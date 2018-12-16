import gym
import numpy as np
import scipy.spatial
import scipy.interpolate
import scipy.sparse
import sklearn

from Challenge_1.util import Discretizer
from Challenge_1.util.state_preprocessing import convert_state_to_sin_cos, normalize_input, \
    unnormalize_input, reconvert_state_to_angle, get_feature_space_boundaries
import sys


class DynamicProgramming(object):

    def __init__(self, action_space, model: callable, discretizer_state: Discretizer,
                 n_actions: int, discount=.99, theta=1e-9, MC_samples=1, verbose=False):

        self.action_space = action_space
        self.high_action = self.action_space.high
        self.low_action = self.action_space.low

        self.discretizer_state = discretizer_state
        self.n_states = self.discretizer_state.n_bins_per_feature

        self.n_actions = n_actions

        self.state_dim = len(self.n_states)
        self.action_dim = self.action_space.shape[0]

        self.model = model

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

        # a = np.tile(self.actions, self.states.shape[0]).reshape(-1, self.action_dim)
        # b = np.repeat(self.discretizer_state.scale_values(self.states), self.n_actions, axis=0).reshape(-1,
        #                                                                                                 self.state_dim)
        # self.tri = scipy.spatial.Delaunay(np.concatenate([a, b], axis=1))

        self.distribution = np.zeros([self.states.shape[0] * self.n_actions] + state_space)

        self.verbose = verbose

        self.transitions, self.reward = self._compute_transition_and_reward_matrices(n_samples=MC_samples)

    def run(self, max_iter=100000):
        raise NotImplementedError

    def _look_ahead(self):
        values_prime = self.value_function[tuple(self.transitions.T)]
        values_new = self.reward + self.discount * values_prime

        return values_new.reshape(-1, self.n_actions)

    def _look_ahead_stochastic(self):

        values_prime = np.sum(np.sum(self.value_function * self.distribution, axis=2), axis=1)
        values_new = self.reward + self.discount * values_prime

        return values_new.reshape(-1, self.n_actions)

    def _compute_transition_and_reward_matrices(self, n_samples):
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

        if self.verbose:
            # this variable is used to show the current progress after 10% blocks
            step_counter = states.shape[0] // 10

            print('Progress ', end="")

            # https://stackoverflow.com/questions/3160699/python-progress-bar
            toolbar_width = 10

            # setup toolbar
            sys.stdout.write("[%s]" % (" " * toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['

        mini_batch_size = 10000

        # predict for each sampled action
        for i in range(states.shape[1]):

            actions_full = actions.reshape(-1, self.action_dim)
            for z in range(0, states.shape[0] * self.n_actions, mini_batch_size):
                # get all samples for all states and one action
                s = np.repeat(states, self.n_actions, axis=0)
                s = s[z:z + mini_batch_size, i, :]

                sel_actions = actions_full[z:z + mini_batch_size]

                state_prime_pred, reward_pred = self.model(s, sel_actions)

                # fill in the state prime prediction and reward
                state_prime[z:z + mini_batch_size, i, :] = state_prime_pred
                reward[i, z:z + mini_batch_size] = reward_pred.flatten()

                if self.verbose and z % step_counter == 0:
                    sys.stdout.write("=")
                    sys.stdout.flush()

            if self.verbose:
                sys.stdout.write("]\n")
                print(']')

        if n_samples == 1:
            # Deterministic case:
            # compute average state transition and reward.
            state_prime = np.mean(state_prime, axis=1)
            state_prime = self.discretizer_state.discretize(state_prime)
        else:
            # TODO stochastic case: return dist over actions and reward
            # Maybe compute reward based on transition probability
            for i in range(state_prime.shape[1]):
                state_prime[:, i, :] = self.discretizer_state.discretize(state_prime[:, i, :])

            for i, elem in enumerate(state_prime):
                u, c = np.unique(elem, return_counts=True, axis=0)
                self.distribution[i][tuple(u.astype(np.int32).T)] = c

        self.distribution /= n_samples
        value = np.mean(reward, axis=0)

        return state_prime, value
