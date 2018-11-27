import gym
import numpy as np

from Challenge_1.util import Discretizer


class DynamicProgramming(object):

    def __init__(self, env: gym.Env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=.99, theta=1e-9):
        """
        :param env:
        :param dynamics_model:
        :param reward_model:
        :param discretizer_state:
        :param discretizer_action:
        :param discount:
        :param theta:
        """

        self.env = env
        self.discretizer_state = discretizer_state
        self.discretizer_action = discretizer_action

        self.n_states = self.discretizer_state.n_bins
        self.n_actions = self.discretizer_action.n_bins

        self.state_dim = self.env.observation_space.shape[0]

        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

        self.discount = discount
        self.theta = theta

        # state_space stores the shape of the state
        state_space = [self.n_states] * self.state_dim

        # np indices returns all possible permutations
        self.states = np.indices(state_space).reshape(self.state_dim, -1).T
        self.actions = np.array(range(0, self.n_actions))
        self.policy = np.random.choice(self.n_actions, size=state_space)
        self.value_function = np.zeros(state_space)

        self.high_state = self.env.observation_space.high
        self.high_action = self.env.action_space.high

        self.low_state = self.env.observation_space.low
        self.low_action = self.env.action_space.low

    def run(self, max_iter=100000):
        raise NotImplementedError

    def _look_ahead(self):
        # scale states to stay within state space
        states = self.discretizer_state.scale_values(self.states)
        states = np.repeat(states, self.n_actions, axis=0)

        # scale actions to stay within action space
        actions = self.actions.reshape(-1, self.env.action_space.shape[0])
        actions = self.discretizer_action.scale_values(actions).flatten()
        actions = np.tile(actions, self.n_states ** self.state_dim)

        # create state-action pairs and use models
        s_a = np.concatenate([states, actions.reshape(-1, self.env.action_space.shape[0])], axis=1)
        state_prime = self.dynamics_model.predict(s_a)
        reward = self.reward_model.predict(s_a)

        # clip to avoid being outside of allowed state space
        state_prime = np.clip(state_prime, self.low_state, self.high_state)
        state_prime_dis = self.discretizer_state.discretize(state_prime)

        # Calculate the expected values of next state
        values_prime = self.value_function[tuple(state_prime_dis.T)]
        values_new = reward + self.discount * values_prime

        return values_new.reshape(-1, self.n_actions)
