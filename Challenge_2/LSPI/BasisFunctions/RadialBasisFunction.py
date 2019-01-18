import numpy as np

from Challenge_2.LSPI.BasisFunctions.BasisFunction import BasisFunction


class RadialBasisFunction(BasisFunction):
    def __init__(self, input_dim, means, n_actions, beta=4):
        """

        :param means: list of rbf centers
        :param beta: hyperparameter for controlling width of gaussians
        """
        super().__init__(input_dim, n_actions)
        self.beta = beta
        self.means = means

    def calc_features(self, observations):
        diff = (self.means[:, None, :] - observations[None, :, :]) / self.beta
        return np.exp(-.5 * np.sum(diff ** 2, axis=2))

    def size(self):
        return (len(self.means) + 1) * self.n_actions

    def get_offset(self, actions):
        return (len(self.means) + 1) * actions
