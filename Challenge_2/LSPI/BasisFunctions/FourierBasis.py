import numpy as np

from Challenge_2.LSPI.BasisFunctions.BasisFunction import BasisFunction


class FourierBasis(BasisFunction):
    def __init__(self, input_dim, frequency, n_actions):
        """
        :param frequency: frequency of Fourier features
        """
        super().__init__(input_dim, n_actions)
        self.frequency = frequency

    def calc_features(self, observations):
        return np.cos(2. * np.pi * self.frequency[None, None, :] * observations[:, :, None]) \
            .reshape(observations.shape[0], self.input_dim * len(self.frequency)).T

    def size(self):
        return (len(self.frequency) * self.input_dim + 1) * self.n_actions

    def get_offset(self, actions):
        return (len(self.frequency) * self.input_dim + 1) * actions
