import numpy as np
from scipy.ndimage.interpolation import shift


class FourierBasis(object):
    def __init__(self, input_dim, widths, n_actions):
        """

        :param input_dim: size of state dimension
        :param widths: list of rbf centers
        :param n_actions: number of output actions
        :param beta: hyperparameter for controlling width of gaussians
        """
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.widths = widths

    def __call__(self, observation, action_idx=None):
        """
        returns fourier features for state action pair. If action is None, values for all actions are returned.
        :param observation: ndarray [batch_size x state_dim]
        :param action_idx: ndarray [batch_size x action_dim]
        :return:
        """
        fourier = self.calc_fourier(observation)

        if action_idx is None:
            # return features for all actions

            phi = np.zeros((observation.shape[0], self.n_actions, self.size(),))

            offset = (len(self.widths) * self.input_dim + 1) * np.arange(0, self.n_actions)

            # write features to the beginning of array for all observations and actions
            phi[:, :, 0] = 1.
            phi = np.transpose(phi, [1, 0, 2])
            phi[:, :, 1:1 + len(fourier)] = fourier.T
            phi = np.transpose(phi, [1, 0, 2])

            # shift elements according to offset for all observations
            phi = phi[:, np.arange(self.n_actions)[:, None], -offset[:, None] + np.arange(self.size())]

        else:
            # return features for one specified action

            phi = np.zeros((observation.shape[0], self.size(),))
            offset = (self.widths.shape[0] + 1) * action_idx.astype(np.int32)

            # write features to the beginning of array for all observations action pairs
            phi[:, 0] = 1.
            phi[:, 1:1 + len(fourier)] = fourier.T

            # shift elements according to offset for all observations
            phi = phi[np.arange(len(observation))[:, None], -offset[:, None] + np.arange(self.size())]

        return phi

    def calc_fourier(self, observations):
        return np.cos(2. * np.pi * self.widths[None, None, :] * observations[:, :, None]) \
            .reshape(observations.shape[0], self.input_dim * len(self.widths)).T

    def size(self):
        return (len(self.widths) * self.input_dim + 1) * self.n_actions
