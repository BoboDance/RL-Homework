import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from Challenge_1.Algorithms.DynamicProgramming import DynamicProgramming
from Challenge_1.util.Discretizer import Discretizer
import logging


class ValueIteration(DynamicProgramming):

    def __init__(self, action_space, model: callable, discretizer_state: Discretizer,
                 n_actions: int, discount=.99, theta=1e-9, MC_samples=1, verbose=False):

        super(ValueIteration, self).__init__(action_space, model, discretizer_state,
                                             n_actions, discount, theta, MC_samples, verbose)

    def run(self, max_iter=100000):

        for i in range(int(max_iter)):
            delta = 0

            # compute value of state with lookahead
            Q = self._look_ahead_stochastic() if self.MC_samples != 1 else self._look_ahead()
            best_value = np.amax(Q, axis=1)

            # get value of all states
            values = self.value_function[tuple(self.states.T)]

            # update value function with new best value
            self.value_function = best_value.reshape(self.value_function.shape)

            delta = np.maximum(delta, np.abs(best_value - values))

            logging.info("Value iteration step: {:6d} -- mean delta: {:4.9f} -- max delta {:4.9f} -- min delta {:4.9f} "
                         .format(i, delta.mean(), delta.max(), delta.min()))

            if np.all(delta <= self.theta):
                print('Value iteration finished in {} iterations.'.format(i + 1))
                break

            if i % 15 == 0 and len(self.value_function.shape) == 2 and i <= 45:
                plt.imshow(self.value_function)
                plt.colorbar()
                plt.xlabel(r'$\.{\theta}$')
                plt.ylabel(r'$\theta$')
                plt.title("Value iteration - value function - iteration {}".format(i))
                plt.show()

        # Create policy in order to use optimal value function
        self.policy = self.actions[
            np.argmax(self._look_ahead_stochastic() if self.MC_samples != 1 else self._look_ahead(), axis=1)].reshape(
            self.policy.shape)

        np.save('./policy_VI', self.policy)
