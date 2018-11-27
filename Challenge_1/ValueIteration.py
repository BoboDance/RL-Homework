import time

import gym
import numpy as np

from Challenge_1.Discretizer import Discretizer


class ValueIteration(object):

    def __init__(self, env: gym.Env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=.99, theta=1e-3):

        self.env = env
        self.discretizer_state = discretizer_state
        self.discretizer_action = discretizer_action

        self.n_states = self.discretizer_state.n_bins
        self.n_actions = self.discretizer_action.n_bins

        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

        self.discount = discount
        self.theta = theta

        state_space = [self.n_states] * self.env.observation_space.shape[0]

        self.states = np.indices(state_space).reshape(self.env.observation_space.shape[0], -1).T
        self.actions = np.array(range(0, self.n_actions))
        self.policy = np.random.choice(self.n_actions, size=state_space)
        self.value_function = np.zeros(state_space)

        self.high_state = self.env.observation_space.high
        self.high_action = self.env.action_space.high

        self.low_state = self.env.observation_space.low
        self.low_action = self.env.action_space.low

    def run(self, max_iter=100000):

        for i in range(int(max_iter)):
            start = time.time()
            delta = 0

            # compute value of state with lookahead
            best_value = np.amax(self._look_ahead(), axis=1)

            # get value of all states
            values = self.value_function[tuple(self.states.T)]

            delta = np.maximum(delta, np.abs(best_value - values))

            # update value function with new best value
            self.value_function = best_value.reshape(self.value_function.shape)

            print("Value iteration step: {:6d} -- mean delta: {:4.4f} -- max delta {:4.4f} -- min delta {:4.4f} "
                  "-- time taken: {:2.4f}s".format(i, delta.mean(), delta.max(), delta.min(), time.time() - start))

            if np.all(delta < self.theta):
                print('Value iteration finished in {} iterations.'.format(i + 1))
                break

        # Create policy in order to use optimal value function
        self.policy = np.argmax(self._look_ahead(), axis=1).reshape(self.policy.shape)

        np.save('./policy_VI', self.policy)

    def _look_ahead(self):

        # scale states to stay within action space
        # states = self.states - (self.n_states - 1) / 2
        # states = states / ((self.n_states - 1) / (2 * self.high_state))
        states = self.discretizer_state.scale_values(self.states)
        states = np.repeat(states, self.n_actions, axis=0)

        actions = np.tile(self.actions, self.n_states ** self.env.observation_space.shape[0])

        # create state-action pairs and use models
        s_a = np.concatenate([states, actions.T.reshape(-1, 1)], axis=1)
        state_prime = self.dynamics_model.predict(s_a)
        reward = self.reward_model.predict(s_a)

        # clip to avoid being outside of allowed state space
        state_prime = np.clip(state_prime, self.low_state, self.high_state)
        state_prime_dis = self.discretizer_state.discretize(state_prime)

        # Calculate the expected values of next state
        values_prime = self.value_function[tuple(state_prime_dis.T)]
        values_new = reward + self.discount * values_prime

        return values_new.reshape(-1, self.n_actions)
