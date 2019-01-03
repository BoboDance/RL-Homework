import numpy as np
from scipy.ndimage.interpolation import shift


class RBF(object):
    def __init__(self, input_dim, means, n_actions, beta=4):
        """

        :param input_dim: size of state dimension
        :param means: list of rbf centers
        :param n_actions: number of output actions
        :param beta: hyperparameter for controlling width of gaussians
        """
        self.input_dim = input_dim
        self.beta = beta
        self.n_actions = n_actions
        self.means = means

    def __call__(self, observation, action_idx=None):
        """
        returns rbf features for state action pair. If action is None, values for all actions are returned.
        :param observation: ndarray [batch_size x state_dim]
        :param action_idx: ndarray [batch_size x action_dim]
        :return:
        """
        rbf = self._calc_rbf(observation)

        if action_idx is None:
            # return features for all actions

            phi = np.zeros((observation.shape[0], self.n_actions, self.size(),))

            offset = (self.means.shape[0] + 1) * np.arange(0, self.n_actions)

            # write features to the beginning of array for all observations and actions
            phi[:, :, 0] = 1.
            phi = np.transpose(phi, [1, 0, 2])
            phi[:, :, 1:1 + len(rbf)] = rbf.T
            phi = np.transpose(phi, [1, 0, 2])

            # shift elements according to offset for all observations
            phi = phi[:, np.arange(self.n_actions)[:, None], -offset[:, None] + np.arange(self.size())]

        else:
            # return features for one specified action

            phi = np.zeros((observation.shape[0], self.size(),))
            offset = (self.means.shape[0] + 1) * action_idx.astype(np.int32)

            # write features to the beginning of array for all observations action pairs
            phi[:, 0] = 1.
            phi[:, 1:1 + len(rbf)] = rbf.T

            # shift elements according to offset for all observations
            phi = phi[np.arange(len(observation))[:, None], -offset[:, None] + np.arange(self.size())]

        return phi

    def _calc_rbf(self, observations):
        diff = (self.means[:, None, :] - observations[None, :, :]) ** 2
        return np.exp(-self.beta * np.sum(diff, axis=2))

    def size(self):
        return (len(self.means) + 1) * self.n_actions
