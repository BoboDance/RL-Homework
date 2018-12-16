import gym
import numpy as np


class DataGenerator(object):

    def __init__(self, env):

        self.env = env

        # get the number of available action from the environment
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        # logging.debug('State Dimension: %d' % self.state_dim)
        # logging.debug('Number of Actions: %d' % self.n_actions)

    def get_samples(self, n_samples=None):
        """
        returns n samples in the form s', s, a, r
        :param n_samples:
        :return: s', s, a, r
        """

        # sample dataset with random actions
        states = np.zeros((n_samples, self.state_dim))
        states_prime = np.zeros((n_samples, self.state_dim))
        actions = np.zeros((n_samples,))
        rewards = np.zeros((n_samples,))

        i = 0
        while i < n_samples:
            done = False
            state = self.env.reset()

            while not done:
                # sample and apply action
                action = self.env.action_space.sample()
                state_prime, reward, done, _ = self.env.step(action)

                if i < n_samples:
                    # save outcomes
                    states_prime[i] = np.array(state_prime)
                    states[i] = np.array(state)
                    actions[i] = action
                    rewards[i] = reward

                # increment counter
                state = state_prime
                i += 1

        return states_prime, states, actions, rewards
