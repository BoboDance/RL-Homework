import gym
import numpy as np

from Challenge_1.util import Discretizer


class DynamicProgramming(object):

    def __init__(self, env: gym.Env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=.99, theta=1e-9, use_MC=False, MC_samples=1):
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
        self.high_state = self.env.observation_space.high
        self.high_action = self.env.action_space.high
        self.low_state = self.env.observation_space.low
        self.low_action = self.env.action_space.low

        self.discretizer_state = discretizer_state
        self.discretizer_action = discretizer_action

        self.n_states = self.discretizer_state.n_bins
        self.n_actions = self.discretizer_action.n_bins

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

        self.discount = discount
        self.theta = theta

        # helper to represent the shape of the state
        state_space = [self.n_states] * self.state_dim

        # np indices returns all possible permutations
        self.states = np.indices(state_space).reshape(self.state_dim, -1).T
        self.actions = np.linspace(self.low_action, self.high_action, self.n_actions)
        self.policy = np.random.choice(self.actions, size=state_space)
        self.value_function = np.zeros(state_space)

        self.use_MC = use_MC

        if use_MC:
            print()
            self.state_prime, self.reward = self.compute_transition_and_reward_matrices(n_samples=MC_samples)

    def run(self, max_iter=100000):
        raise NotImplementedError

    def _look_ahead(self):

        # TODO This part is not required when using the MC approach
        # ------------------------------------------------------------------
        # scale states to stay within state space
        states = self.discretizer_state.scale_values(self.states)
        states = np.repeat(states, self.n_actions, axis=0)

        # scale actions to stay within action space
        actions = self.actions.reshape(-1, self.action_dim)
        actions = self.discretizer_action.scale_values(actions).flatten()
        actions = np.tile(actions, states.shape[0])

        # create state-action pairs and use models
        s_a = np.concatenate([states, actions.reshape(-1, self.action_dim)], axis=1)
        state_prime = self.dynamics_model.predict(s_a)
        reward = self.reward_model.predict(s_a)

        # state_prime = np.zeros_like(states)
        # reward = np.zeros_like(actions)
        #
        # self.env.reset()
        # for j, s in enumerate(s_a):
        #     self.env.env.state = s[:-1]
        #     # state_prime[j], reward[j], _, _ = self.env.step(np.array([s[-1]]))
        #     state_prime[j], _, _, _ = self.env.step(np.array([s[-1]]))

        # clip to avoid being outside of allowed state space
        state_prime = np.clip(state_prime, self.low_state, self.high_state)
        state_prime_dis = self.discretizer_state.discretize(state_prime)

        # Calculate the expected values of next state for all actions
        values_prime = self.value_function[tuple(state_prime_dis.T)]
        values_new = reward + self.discount * values_prime

        return values_new.reshape(-1, self.n_actions)

    def _look_ahead_MC(self):
        values_prime = self.value_function[tuple(self.state_prime.T)]
        values_new = self.reward + self.discount * values_prime

        return values_new.reshape(-1, self.n_actions)

    def compute_transition_and_reward_matrices(self, n_samples):
        # create n samples within the bins an average in order to get better representation
        states = self.discretizer_state.scale_values_stochastic(self.states, n_samples=n_samples)
        actions = np.tile(self.actions, states.shape[0])

        # create state-action pairs and use models
        state_prime = np.zeros((states.shape[0] * self.n_actions, states.shape[1], states.shape[2]))
        reward = np.zeros((states.shape[1], actions.shape[0]))

        # predict for each sampled action
        for i in range(states.shape[1]):
            # get all samples for all states and one action
            s = states[:, i, :]
            s = np.repeat(s, self.n_actions, axis=0)
            s_a = np.concatenate([s, actions.reshape(-1, self.action_dim)], axis=1)
            state_prime[:, i, :] = self.dynamics_model.predict(s_a)
            reward[i] = self.reward_model.predict(s_a).flatten()

        # Deterministic case:
        # compute average state transition and reward.
        # TODO stochastic case: return dist over actions and reward
        # Maybe compute reward based on transition probability
        state_prime = np.mean(state_prime, axis=1)
        state_prime = self.discretizer_state.discretize(state_prime)
        value = np.mean(reward, axis=0)

        return state_prime, value
