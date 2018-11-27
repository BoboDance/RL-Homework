import time

import gym
import numpy as np

from Challenge_1.Algorithms.DynamicProgramming import DynamicProgramming
from Challenge_1.util import Discretizer


class PolicyIteration(DynamicProgramming):

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
        super(PolicyIteration, self).__init__(env, dynamics_model, reward_model, discretizer_state, discretizer_action,
                                              discount, theta)

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
            actions = self.policy
            actions = self.discretizer_action.scale_values(actions)

            # scale states to stay within action space
            states = self.discretizer_state.scale_values(self.states)

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

            # adjust value function, based on results from action
            values = self.value_function[tuple(self.states.T)]
            delta = np.maximum(delta, np.abs(values - values_new))
            self.value_function = values_new.reshape(self.value_function.shape)

            print("Policy evaluation step: {:6d} -- mean delta: {:4.9f} -- max delta {:4.9f} -- min delta {:4.9f} "
                  "-- time taken: {:2.4f}s".format(i, delta.mean(), delta.max(), delta.min(), time.time() - start))

            # Terminate if change is below threshold
            if np.all(delta < self.theta):
                print('Policy evaluation finished in {} iterations.'.format(i + 1))
                break

    def _policy_improvement(self):

        stable = True
        # TODO: time returns a weird value
        start = time.time()

        # Choose action with current policy
        policy_action = self.policy
        policy_action = policy_action.reshape(-1, self.env.action_space.shape[0])

        # Check if current policy_action is actually best by checking all actions
        best_action = np.argmax(self._look_ahead(), axis=1)
        best_action = best_action.reshape(-1, self.env.action_space.shape[0])

        # If better action was found
        if np.any(policy_action != best_action):
            stable = False
            # Greedy policy update
            self.policy = best_action.reshape(self.policy.shape)
            print("# of incorrectly selected actions: {}".format(np.count_nonzero(policy_action != best_action)))
            # if np.count_nonzero(policy_action != best_action) < 5:
            #     stable = True

        print(
            "Policy improvement finished -- stable: {} -- time taken: {:2.4f}s".format(stable, time.time() - start))
        return stable
