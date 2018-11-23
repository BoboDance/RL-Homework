import time

import gym
import numpy as np

from Challenge_1 import Discretizer


class PolicyIteration(object):

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

        self.policy = np.ones(state_space + [self.n_actions]) / self.n_actions
        self.value_function = np.zeros(state_space)

        self.high_state = self.env.observation_space.high
        self.high_action = self.env.action_space.high

        # -----------------------------------------------------------------------
        # dict version

        # f = lambda x: defaultdict(lambda: x)

        # self.value_function = defaultdict(float)
        # for _ in range(self.env.observation_space.shape[0] - 1):
        #     self.value_function = f(self.value_function)
        #
        # self.init_dicts(self.value_function, self.state_dim)
        #
        # # Random policy with equally likely actions
        # self.policy = defaultdict(lambda: 1. / self.n_actions)
        #
        # for _ in range(self.env.observation_space.shape[0]):
        #     self.policy = f(self.policy)
        #
        # self.init_dicts(self.policy, self.state_dim)

    def _policy_evaluation(self, max_iter=1000):

        start = time.clock()
        for i in range(max_iter):

            delta = 0
            # for state_0, d1 in self.policy.items():
            for state_0 in range(self.state_dim):

                # for state_1, d2 in d1.items():
                #     for state_2 in d2.keys():
                for state_1 in range(self.state_dim):
                    for state_2 in range(self.state_dim):

                        # current_value = 0

                        state_concat = np.array([state_0, state_1, state_2])
                        # shift to be in the actual state range and not 0 to 2*high
                        state_concat = state_concat - self.high_state

                        # Try out all possible actions for this state
                        # for action, prob in self.policy[state_0][state_1][state_2].items():

                        # for action, prob in enumerate(self.policy[state_0, state_1, state_2]):
                        prob = self.policy[state_0, state_1, state_2]
                        action = np.argmax(prob)
                        action = action - self.high_action

                        # compute actions based on models without interaction with env
                        s_a = np.concatenate([state_concat, action])
                        s_a = s_a.reshape(-1, self.env.observation_space.shape[0] + self.env.action_space.shape[0])
                        state_prime = self.dynamics_model.predict(s_a)
                        reward = self.reward_model.predict(s_a)

                        # clip to avoid being outside of allowed state space
                        state_prime = np.clip(state_prime, self.env.observation_space.low,
                                              self.env.observation_space.high).flatten()
                        state_prime_dis = self.discretizer_state.discretize(state_prime)

                        # Calculate the expected value of next state
                        value_prime = self.value_function[
                            state_prime_dis[0], state_prime_dis[1], state_prime_dis[2]]
                        current_value = prob[action.astype(np.int32)] * (reward + self.discount * value_prime)

                        # Change of value function
                        value = self.value_function[state_0, state_1, state_2]
                        delta = np.maximum(delta, np.abs(value - current_value))
                        self.value_function[state_0, state_1, state_2] = current_value

            start = time.clock() - start
            print(
                "Policy eval step: {:d} -- delta: {:4.4f} -- time taken: {:d}:{:2d}".format(i, delta[0], int(start) // 60, int(start) % 60))

            # Terminate if change is below threshold
            if delta < self.theta:
                print('Policy evaluation finished in {} iterations.'.format(i + 1))
                break

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

    def policy_improvement(self):
        # policy improvement
        # for state_0, d1 in self.policy.items():
        #     for state_1, d2 in d1.items():
        #         for state_2 in d2.keys():
        stable = True
        start = time.clock()

        for state_0 in range(self.state_dim):
            for state_1 in range(self.state_dim):
                for state_2 in range(self.state_dim):

                    state_concat = np.array([state_0, state_1, state_2])

                    # Choose action with current policy
                    policy_action = np.argmax(self.policy[state_0, state_1, state_2]) - self.high_action

                    # Check if current action is actually best
                    best_action = np.argmax(self._get_action_dist(state_concat))
                    best_action_shifted = best_action - self.high_action

                    # If action didn't change
                    if policy_action != best_action_shifted:
                        stable = False
                        # Greedy policy update
                        self.policy[state_0, state_1, state_2] = np.eye(self.n_actions)[best_action]
                        # for k, v in self.policy[state_0][state_1,state_2].items():
                        #     if k != best_action:
                        #         self.policy[state_0][state_1][state_2][k] = 0
                        #     else:
                        #         self.policy[state_0][state_1][state_2][k] = 1

        start = time.clock() - start
        print(
            "Policy improvement finished -- stable: {} -- time taken: {}:{:2d}".format(stable, int(start) // 60,
                                                                                       int(start) % 60))
        return stable

    def run(self, max_iter=1000):

        stable = False
        i = 0

        while not stable:
            print("Policy iteration step: {}".format(i))

            # determines value function
            self._policy_evaluation(max_iter=max_iter)
            stable = self.policy_improvement()
            i += 1

            # policy iteration converged
            if stable:
                print('Evaluated {} policies and found stable policy'.format(i + 1))
                # import json
                # with open('./result.json', 'w') as fp:
                #     json.dump(self.policy, fp)
                # break
                np.save('./policy', self.policy)

    # def load_policy(self):
    #     import json
    #     with open('./result.json', 'w') as fp:
    #         json.dump(self.policy, fp)
    #     break

    # def init_dicts(self, dict, n):
    #     for i in range(n):
    #         d = dict[i]
    #         if isinstance(d, defaultdict):
    #             self.init_dicts(d, n)
