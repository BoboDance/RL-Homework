import time

import gym
import numpy as np

from Challenge_1.Discretizer import Discretizer


class ValueIteration(object):

    def __init__(self, env: gym.Env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=1., theta=1e-3):

        self.env = env
        self.discretizer_state = discretizer_state
        self.discretizer_action = discretizer_action

        self.state_dim = self.discretizer_state.n_bins
        self.n_actions = self.discretizer_action.n_bins

        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

        self.discount = discount
        self.theta = theta

        state_space = [self.state_dim] * self.env.observation_space.shape[0]

        self.policy = np.zeros(state_space + [self.n_actions])
        self.value_function = np.zeros(state_space)

        self.high_state = self.env.observation_space.high
        self.high_action = self.env.action_space.high

    def run(self, max_iter=1000):

        start = time.clock()

        for i in range(int(max_iter)):

            delta = 0

            for state_0 in range(self.state_dim):
                for state_1 in range(self.state_dim):
                    for state_2 in range(self.state_dim):
                        state_concat = np.array([state_0, state_1, state_2])
                        best_action_value = np.argmax(self._get_action_dist(state_concat))

                        delta = np.maximum(delta,
                                           np.abs(self.value_function[state_0, state_1, state_2] - best_action_value))

                        self.value_function[state_0, state_1, state_2] = best_action_value

            start = time.clock() - start
            print(
                "Value iteration step: {:d} -- delta: {:4.4f} -- time taken: {:d}:{:2d}".format(i, delta,
                                                                                                int(start) // 60,
                                                                                                int(start) % 60))

            # print(self.value_function)

            if delta < self.theta:
                print('Value iteration finished in {} iterations.'.format(i + 1))
                break

        # Create policy in order to use optimal value function

        for state_0 in range(self.state_dim):
            for state_1 in range(self.state_dim):
                for state_2 in range(self.state_dim):
                    state_concat = np.array([state_0, state_1, state_2])
                    # Select best action based on the highest state-action value
                    best_action = np.argmax(self._get_action_dist(state_concat))
                    self.policy[state_0, state_1, state_2, best_action] = 1.0

    def _get_action_dist(self, state):

        # action_prob = defaultdict(int)
        action_prob = np.zeros(self.n_actions)
        state_shifted = state - self.high_state

        # print(self.env.observation_space.high)

        for action, _ in enumerate(self.policy[state[0], state[1], state[2]]):
            # compute actions based on models without interaction with env

            action_shifted = action - self.high_action

            s_a = np.concatenate([state_shifted, action_shifted]).reshape(1, -1)
            state_prime = self.dynamics_model.predict(s_a).flatten()
            # print(state_prime)
            state_prime = np.clip(state_prime, self.env.observation_space.low,
                                  self.env.observation_space.high)
            # print(state_prime)
            state_prime_dis = self.discretizer_state.discretize(state_prime)
            state_prime_dis = state_prime_dis.astype(np.int32)
            # print(state_prime_dis)

            reward = self.reward_model.predict(s_a).flatten()

            value = self.value_function[state_prime_dis[0], state_prime_dis[1], state_prime_dis[2]]
            action_prob[action] += (reward + self.discount * value)

        # normalize to sum up to one, not really necessary, we only look for argmax
        action_prob /= np.sum(action_prob)
        return action_prob
