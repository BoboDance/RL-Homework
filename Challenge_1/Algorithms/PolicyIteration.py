import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from Challenge_1.Algorithms.DynamicProgramming import DynamicProgramming
from Challenge_1.util import Discretizer


class PolicyIteration(DynamicProgramming):

    def __init__(self, env: gym.Env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=.99, theta=1e-9, use_MC=True, MC_samples=1):
        """

        :param env:
        :param dynamics_model:
        :param reward_model:
        :param discretizer_state:
        :param discretizer_action:
        :param discount:
        :param theta:
        """
        super(PolicyIteration, self).__init__(env, dynamics_model, reward_model, discretizer_state, discretizer_action,
                                              discount, theta, use_MC, MC_samples)

    def run(self, max_iter=100000):

        stable = False
        i = 0

        while not stable:
            print("Policy iteration step: {}".format(i))

            self._policy_evaluation(max_iter=max_iter)
            stable = self._policy_improvement()
            i += 1

            # policy iteration converged
            if stable:
                print('Evaluated {} policies and found stable policy'.format(i + 1))
                np.save('./policy_PI', self.policy)

    def _policy_evaluation(self, max_iter=100000):

        for i in range(max_iter):
            start = time.time()
            delta = 0

            # choose best action for each state
            # scale actions to stay within action space
            # actions = self.policy
            # actions = actions.reshape(-1, self.action_dim)
            # actions = self.discretizer_action.scale_values(actions)

            # scale states to stay within action space
            # states = self.discretizer_state.scale_values(self.states)

            # create state-action pairs and use models
            # s_a = np.concatenate([states, actions], axis=1)
            # state_prime = self.dynamics_model.predict(s_a)
            # reward = self.reward_model.predict(s_a)

            action_idx = np.searchsorted(self.actions, self.policy).flatten()
            state_prime = self.state_prime.reshape(self.policy.shape + (self.n_actions,) + (self.state_dim,))
            state_prime = state_prime[tuple(self.states.T)][np.arange(0, len(action_idx)), action_idx, :]

            reward = self.reward.reshape(self.policy.shape + (self.n_actions,))
            reward = reward[tuple(self.states.T)][np.arange(0, len(action_idx)), action_idx]

            # er = np.zeros_like(states)
            # abc = np.zeros_like(actions)
            #
            # self.env.reset()
            # for j, s in enumerate(s_a):
            #     self.env.env.state = s[:-1]
            #     # state_prime[j], reward[j], _, _ = self.env.step(np.array([s[-1]]))
            #     er[j], abc[j], _, _ = self.env.step(np.array([s[-1]]))

            # clip to avoid being outside of allowed state space
            # state_prime = np.clip(state_prime, self.low_state, self.high_state)
            # state_prime_dis = self.discretizer_state.discretize(state_prime)

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

                if len(self.value_function.shape) == 2:
                    plt.matshow(self.value_function)
                    plt.colorbar()
                    plt.title("Value function")
                    plt.show()

                break

    def _policy_improvement(self):

        stable = True
        # TODO: time returns a weird value
        start = time.time()

        # Choose action with current policy
        policy_action = self.policy.flatten()
        # policy_action = policy_action.reshape(-1, self.action_dim)

        # Check if current policy_action is actually best by checking all actions
        value_map = self._look_ahead_MC() if self.use_MC else self._look_ahead()
        best_action = self.actions[np.argmax(value_map, axis=1)]
        # best_action = np.argmax(self._look_ahead(), axis=1)
        # best_action = best_action.reshape(-1, self.action_dim)

        # If better action was found
        if np.any(policy_action != best_action):
            stable = False
            # Greedy policy update
            self.policy = best_action.reshape(self.policy.shape)
            print("# of incorrectly selected actions: {}".format(np.count_nonzero(policy_action != best_action)))
            # if np.count_nonzero(policy_action != best_action) < 50:
            #     stable = True

        print(
            "Policy improvement finished -- stable: {} -- time taken: {:2.4f}s".format(stable, time.time() - start))

        if len(self.policy.shape) == 2:
            plt.matshow(self.value_function)
            plt.colorbar()
            plt.title("Policy")
            plt.show()

        return stable
