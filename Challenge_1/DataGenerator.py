import gym
import numpy as np


class DataGenerator():

    def __init__(self, env_name, seed):
        self.env_name = env_name
        self.seed = seed

        self.env = gym.make(self.env_name)
        self.env.seed(self.seed)
        np.random.seed(self.seed)

        if env_name == "Pendulum-v0":
            self.n_samples = 10000
        elif env_name == "Qube-v0":
            self.n_samples = 25000
        else:
            raise NotImplementedError("Unsupported Environment")

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

        if n_samples is not None:
            self.n_samples = n_samples

        # sample dataset with random actions
        states = np.zeros((self.n_samples, self.state_dim))
        states_prime = np.zeros((self.n_samples, self.state_dim))
        actions = np.zeros((self.n_samples,))
        rewards = np.zeros((self.n_samples,))

        i = 0
        while i < self.n_samples:
            state = self.env.reset()
            done = False

            while not done and i < self.n_samples:
                # sample and apply action
                action = self.env.action_space.sample()
                state_prime, reward, done, _ = self.env.step(action)

                # save outcomes
                states_prime[i] = np.array(state_prime)
                states[i] = np.array(state)
                actions[i] = action
                rewards[i] = reward

                # increment counter
                state = state_prime
                i += 1

        return states_prime, states, actions, rewards
