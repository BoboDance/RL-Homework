import numpy as np

from Challenge_2.LSPI.BasisFunctions.BasisFunction import BasisFunction


class FourierBasis(BasisFunction):
    def __init__(self, input_dim, n_features, n_actions):
        """
        :param n_features: frequency of Fourier features
        """
        super().__init__(input_dim, n_actions)
        # self.frequency = frequency
        self.n_features = n_features
        self.frequency = np.random.normal(0, 1, size=(n_features, input_dim))
        self.bandwidth = 5
        self.shift = np.random.uniform(-np.pi, np.pi, n_features)
        # self.shift = np.random.uniform(0, 2 * np.pi, n_features)

    def calc_features(self, observations):
        return np.sin(
            np.sum(self.frequency[:, None, :] * observations[None, :, :], axis=2) / self.bandwidth + self.shift[:,
                                                                                                     None])
        # return np.sqrt(2 / self.n_features) * np.cos(
        #     np.sum(self.frequency[:, None, :] * observations[None, :, :], axis=2) + self.shift[:, None])

    def size(self):
        return (len(self.frequency) + 1) * self.n_actions

    def get_offset(self, actions):
        return (len(self.frequency) + 1) * actions
