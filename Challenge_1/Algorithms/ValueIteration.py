import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from Challenge_1.Algorithms.DynamicProgramming import DynamicProgramming
from Challenge_1.util.Discretizer import Discretizer


class ValueIteration(DynamicProgramming):

    def __init__(self, env: gym.Env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=.99, theta=1e-3, use_MC=False, MC_samples=1,
                 angle_features=[0]):

        super(ValueIteration, self).__init__(env, dynamics_model, reward_model, discretizer_state, discretizer_action,
                                             discount, theta, use_MC, MC_samples, angle_features)

    def run(self, max_iter=100000):

        for i in range(int(max_iter)):

            start = time.time()
            delta = 0

            # compute value of state with lookahead
            value_map = self._look_ahead_MC() if self.use_MC else self._look_ahead()
            best_value = np.amax(value_map, axis=1)

            # get value of all states
            values = self.value_function[tuple(self.states.T)]

            # update value function with new best value
            self.value_function = best_value.reshape(self.value_function.shape)

            delta = np.maximum(delta, np.abs(best_value - values))

            # print("Value iteration step: {:6d} -- mean delta: {:4.9f} -- max delta {:4.9f} -- min delta {:4.9f} "
            #       "-- time taken: {:2.4f}s".format(i, delta.mean(), delta.max(), delta.min(), time.time() - start))

            if np.all(delta <= self.theta):
                # print('Value iteration finished in {} iterations.'.format(i + 1))
                break

            # if i % 15 == 0 and len(self.value_function.shape) == 2:
            #     plt.matshow(self.value_function)
            #     plt.colorbar()
            #     plt.title("Value function")
            #     plt.show()

        # Create policy in order to use optimal value function
        self.policy = self.actions[
            np.argmax(self._look_ahead_MC() if self.use_MC else self._look_ahead(), axis=1)].reshape(self.policy.shape)

        # np.save('./policy_VI', self.policy)
