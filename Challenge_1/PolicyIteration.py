import time

import gym
import numpy as np

from Challenge_1 import Discretizer


class PolicyIteration(object):

    def __init__(self, env: gym.Env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=.99, theta=1e-3):
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
        self.discretizer_state = discretizer_state
        self.discretizer_action = discretizer_action

        self.n_states = self.discretizer_state.n_bins
        self.n_actions = self.discretizer_action.n_bins

        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

        self.discount = discount
        self.theta = theta

        # state_space stores the shape of the state
        state_space = [self.n_states] * self.env.observation_space.shape[0]

        # np indices returns all possible permutations
        self.states = np.indices(state_space).reshape(self.env.observation_space.shape[0], -1).T
        self.actions = np.array(range(0, self.n_actions))
        self.policy = np.random.choice(self.n_actions, size=state_space)
        self.value_function = np.zeros(state_space)

        self.high_state = self.env.observation_space.high
        self.low_state = self.env.observation_space.low
        self.high_action = self.env.action_space.high

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
                np.save('./policy_PI.npc', self.policy)

    def _policy_evaluation(self, max_iter=100000):

        # TODO: time returns a weird value
        start = time.time()
        for i in range(max_iter):
            delta = 0

            # choose best action for each state
            # actions = np.argmax(self.policy, axis=self.env.observation_space.shape[0])
            actions = self.policy

            # scale actions to stay within action space
            actions = actions - (self.n_actions - 1) / 2
            actions = actions / ((self.n_actions - 1) / (2 * self.high_action))

            # scale states to stay within action space
            total_range = (self.high_state - self.low_state)
            state_01 = self.states / (self.n_states-1)
            states = (state_01 * total_range) + self.low_state

            # create state-action pairs and use models
            s_a = np.concatenate([states, actions.T.reshape(-1, self.env.action_space.shape[0])], axis=1)
            state_prime = self.dynamics_model.predict(s_a)
            reward = self.reward_model.predict(s_a)

            # clip to avoid being outside of allowed state space
            state_prime = np.clip(state_prime, -self.high_state, self.high_state)
            state_prime_dis = self.discretizer_state.discretize(state_prime)

            # Calculate the expected values of next state
            values_prime = self.value_function[tuple(state_prime_dis.T)]
            values_new = reward + self.discount * values_prime

            # adjust value function, based on results from action
            values = self.value_function[tuple(self.states.T)]
            delta = np.maximum(delta, np.abs(values - values_new))
            self.value_function = values_new.reshape(self.value_function.shape)

            start = time.time() - start
            # print("Policy eval step: {:d} -- delta: {:4.4f} -- time taken: {:d}:{:2d}".format(i, delta[0],
            #                                                                                   int(start) // 60,
            #                                                                                   int(start) % 60))

            print("Policy evaluation step: {:6d} -- mean delta: {:4.4f} -- max delta {:4.4f} -- min delta {:4.4f} "
                  "-- time taken: {:d}:{:2d}".format(i, delta.mean(), delta.max(), delta.min(), int(start) // 60,
                                                     int(start) % 60))

            # Terminate if change is below threshold
            if np.all(delta < self.theta):
                print('Policy evaluation finished in {} iterations.'.format(i + 1))
                break

    def _policy_improvement(self):

        stable = True
        # TODO: time returns a weird value
        start = time.time()

        # Choose action with current policy
        # policy_action = np.argmax(self.policy, axis=self.env.observation_space.shape[0])
        policy_action = self.policy

        # scale policy_action to stay within action space
        # policy_action = policy_action - (self.n_actions - 1) / 2
        # policy_action = policy_action / ((self.n_actions - 1) / (2 * self.high_action))
        policy_action = policy_action.T.reshape(-1, self.env.action_space.shape[0])

        # Check if current policy_action is actually best
        # TODO: Implement the negative shift into the mountain car problem to see if it's working then
        best_action = np.argmax(self._look_ahead(), axis=1).reshape(-1, self.env.action_space.shape[0])

        # scale best_action to stay within action space
        # best_action = best_action - (self.n_actions - 1) / 2
        # best_action = best_action / ((self.n_actions - 1) / (2 * self.high_action))

        # If better action was found
        if np.any(policy_action != best_action):
            stable = False
            # Greedy policy update
            self.policy = best_action.reshape(self.policy.shape)
            print(np.count_nonzero(policy_action != best_action))
            if np.count_nonzero(policy_action != best_action) < 5:
                stable = True

        start = time.time() - start
        print(
            "Policy improvement finished -- stable: {} -- time taken: {}:{:2d}".format(stable, int(start) // 60,
                                                                                       int(start) % 60))
        return stable

    def _look_ahead(self):
        # scale states to stay within action space
        total_range = (self.high_state - self.low_state)
        state_01 = self.states / (self.n_states - 1)
        states = (state_01 * total_range) + self.low_state

        states = np.repeat(states, self.n_actions, axis=0)

        # actions = self.policy[tuple(self.states.T)]
        actions = np.tile(self.actions, self.n_states ** self.env.observation_space.shape[0])

        # create state-action pairs and use models
        s_a = np.concatenate([states, actions.T.reshape(-1, 1)], axis=1)
        state_prime = self.dynamics_model.predict(s_a)
        reward = self.reward_model.predict(s_a)

        # clip to avoid being outside of allowed state space
        state_prime = np.clip(state_prime, -self.high_state, self.high_state)
        state_prime_dis = self.discretizer_state.discretize(state_prime)

        # Calculate the expected values of next state
        values_prime = self.value_function[tuple(state_prime_dis.T)]
        values_new = reward + self.discount * values_prime

        return values_new.reshape(-1, self.n_actions)
