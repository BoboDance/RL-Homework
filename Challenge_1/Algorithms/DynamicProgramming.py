import gym
import numpy as np

from Challenge_1.util import Discretizer
from Challenge_1.util.state_preprocessing import convert_state_to_sin_cos, normalize_input, \
    unnormalize_input, reconvert_state_to_angle, get_feature_space_boundaries
import sys


class DynamicProgramming(object):

    def __init__(self, env: gym.Env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=.99, theta=1e-9, use_MC=False, MC_samples=1,
                 angle_features=[0]):
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

        self.n_states = self.discretizer_state.n_bins_per_feature
        self.n_actions = self.discretizer_action.n_bins_per_feature[0]

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

        self.discount = discount
        self.theta = theta

        # helper to represent the shape of the state
        # state_space = [self.n_states] * self.state_dim
        state_space = self.n_states

        # np indices returns all possible permutations
        self.states = np.indices(state_space).reshape(self.state_dim, -1).T
        self.actions = np.linspace(self.low_action, self.high_action, self.n_actions)
        self.policy = np.random.choice(self.actions, size=state_space)
        self.value_function = np.zeros(state_space)

        self.use_MC = use_MC

        self.angle_features = angle_features

        if use_MC:
            self.state_prime, self.reward = self.compute_transition_and_reward_matrices(n_samples=MC_samples)

    def run(self, max_iter=100000):
        raise NotImplementedError

    def _look_ahead_MC(self):
        values_prime = self.value_function[tuple(self.state_prime.T)]
        values_new = self.reward + self.discount * values_prime

        return values_new.reshape(-1, self.n_actions)

    def compute_transition_and_reward_matrices(self, n_samples):
        # create n samples within the bins an average in order to get better representation
        if n_samples == 1:
            states = self.discretizer_state.scale_values(self.states).reshape(self.states.shape[0], 1,
                                                                              self.states.shape[1])
        else:
            states = self.discretizer_state.scale_values_stochastic(self.states, n_samples=n_samples)

        actions = np.tile(self.actions, states.shape[0])

        # create state-action pairs and use models
        state_prime = np.zeros((states.shape[0] * self.n_actions, states.shape[1], states.shape[2]))
        reward = np.zeros((states.shape[1], actions.shape[0]))

        # print('states.shape[1]', states.shape[1])

        # this variable is used to show the current progress after 10% blocks
        # step_counter = states.shape[1] // 10

        # print('Progress ', end="")

        # https://stackoverflow.com/questions/3160699/python-progress-bar
        # toolbar_width = 10

        # setup toolbar
        # sys.stdout.write("[%s]" % (" " * toolbar_width))
        # sys.stdout.flush()
        # sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

        # predict for each sampled action
        for i in range(states.shape[1]):
            # get all samples for all states and one action
            s = states[:, i, :]
            s = np.repeat(s, self.n_actions, axis=0)

            s = convert_state_to_sin_cos(s, self.angle_features)
            s_a = np.concatenate([s, actions.reshape(-1, self.action_dim)], axis=1)

            # create both boundaries of the feature space
            x_low, x_high = get_feature_space_boundaries(self.env, self.angle_features)

            # normalize the state action pair to the range [0,1]
            s_a = normalize_input(s_a, x_low, x_high)

            # request the next state from our dynamics model
            state_prime_pred = self.dynamics_model.predict(s_a)

            # after predicting the state prime from our dynamic model we should rennormalize it back
            # to it's original from. The sin(angle) replaces the original angle and the cos(angle) is appended to the
            # original state representation
            # unnormalize the state back to it's original state ranges
            state_prime_pred = unnormalize_input(state_prime_pred, x_low[:-1], x_high[:-1])
            # reconvert the angle feature back to a single angle to have a more compact representation
            state_prime_pred = reconvert_state_to_angle(state_prime_pred, self.angle_features)

            # fill in the state prime prediction
            state_prime[:, i, :] = state_prime_pred

            # request the reward prediction of the corresponding reward from our model
            reward[i] = self.reward_model.predict(s_a).flatten()

            # if i % step_counter == 0:
            #     sys.stdout.write("=")
            #     sys.stdout.flush()

        # sys.stdout.write("]\n")

        # print(']')

        # Deterministic case:
        # compute average state transition and reward.
        # TODO stochastic case: return dist over actions and reward
        # Maybe compute reward based on transition probability
        state_prime = np.mean(state_prime, axis=1)
        state_prime = self.discretizer_state.discretize(state_prime)
        value = np.mean(reward, axis=0)

        return state_prime, value
