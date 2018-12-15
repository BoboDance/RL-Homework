import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from Challenge_1.Algorithms.DynamicProgramming import DynamicProgramming
from Challenge_1.util import Discretizer


class PolicyIteration(DynamicProgramming):

    def __init__(self, action_space, model: callable, discretizer_state: Discretizer,
                 n_actions: int, discount=.99, theta=1e-9, MC_samples=1, angle_features=[0],
                 verbose=False):
        """

        :param env:
        :param dynamics_model:
        :param reward_model:
        :param discretizer_state:
        :param discretizer_action:
        :param discount:
        :param theta:
        """
        super(PolicyIteration, self).__init__(action_space, model, discretizer_state,
                                              n_actions, discount, theta, MC_samples, angle_features, verbose)

    def run(self, max_iter=100000):

        stable = False
        i = 0

        while not stable:
            # print("Policy iteration step: {}".format(i))

            self._policy_evaluation(max_iter=max_iter)
            stable = self._policy_improvement()
            i += 1

            # policy iteration converged
            if stable:
                print('Evaluated {} policies and found stable policy'.format(i + 1))
                np.save('./policy_PI', self.policy)

                if len(self.value_function.shape) == 2:
                    plt.matshow(self.value_function)
                    plt.colorbar()
                    plt.title("Value function policy iteration")
                    plt.show()

    def _policy_evaluation(self, max_iter=100000):

        for i in range(max_iter):
            start = time.time()
            delta = 0

            # get index of best action from policy
            action_idx = np.searchsorted(self.actions, self.policy).flatten()

            # select only successor states from transition matrix which are reached based on the policy actions
            state_prime = self.transitions.reshape(self.policy.shape + (self.n_actions,) + (self.state_dim,))
            state_prime = state_prime[tuple(self.states.T)][np.arange(0, len(action_idx)), action_idx, :]

            # select only rewards from reward matrix which are reached based on the policy actions
            reward = self.reward.reshape(self.policy.shape + (self.n_actions,))
            reward = reward[tuple(self.states.T)][np.arange(0, len(action_idx)), action_idx]

            # Calculate the expected values of next state
            values_prime = self.value_function[tuple(state_prime.T)]
            values_new = reward + self.discount * values_prime

            # calculate convergence criterion
            values = self.value_function[tuple(self.states.T)]
            delta = np.maximum(delta, np.abs(values - values_new))

            # adjust value function, based on results from action
            self.value_function = values_new.reshape(self.value_function.shape)

            print("Policy evaluation step: {:6d} -- mean delta: {:4.9f} -- max delta {:4.9f} -- min delta {:4.9f} "
                  "-- time taken: {:2.4f}s".format(i, delta.mean(), delta.max(), delta.min(), time.time() - start))

            # Terminate if change is below threshold
            if np.all(delta <= self.theta):
                print('Policy evaluation finished in {} iterations.'.format(i + 1))
                break

    def _policy_improvement(self):

        stable = True
        start = time.time()

        # Choose action with current policy
        policy_action = self.policy.flatten()

        # Check if current policy_action is actually best by checking all actions
        Q = self._look_ahead()
        best_action = self.actions[np.argmax(Q, axis=1)]

        # If better action was found
        if np.any(policy_action != best_action):
            stable = False
            # Greedy policy update
            self.policy = best_action.reshape(self.policy.shape)
            print("# of incorrectly selected actions: {}".format(np.count_nonzero(policy_action != best_action)))

        print(
            "Policy improvement finished -- stable: {} -- time taken: {:2.4f}s".format(stable, time.time() - start))

        return stable
