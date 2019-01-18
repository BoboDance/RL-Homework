from abc import ABC, abstractmethod

import numpy as np


class BasisFunction(ABC):
    def __init__(self, input_dim, n_actions):
        """

        :param input_dim: size of state dimension
        :param n_actions: number of output actions
        """
        self.input_dim = input_dim
        self.n_actions = n_actions

    def __call__(self, observation, action_idx=None):
        """
        computes features for state action pair. If action is None, values for all actions are returned.
        :param observation: ndarray [batch_size x state_dim]
        :param action_idx: ndarray [batch_size x action_dim]
        :return: features for given states and actions
        """
        features = self.calc_features(observation)

        if action_idx is None:
            # return features for all actions

            phi = np.zeros((observation.shape[0], self.n_actions, self.size(),))

            offset = self.get_offset(np.arange(0, self.n_actions))

            # write features to the beginning of array for all observations and actions
            phi[:, :, 0] = 1.
            phi = np.transpose(phi, [1, 0, 2])
            phi[:, :, 1:1 + len(features)] = features.T
            phi = np.transpose(phi, [1, 0, 2])

            # shift elements according to offset for all observations
            phi = phi[:, np.arange(self.n_actions)[:, None], -offset[:, None] + np.arange(self.size())]

        else:
            # return features for one specified action

            phi = np.zeros((observation.shape[0], self.size(),))
            offset = self.get_offset(action_idx.astype(np.int32))

            # write features to the beginning of array for all observations action pairs
            phi[:, 0] = 1.
            phi[:, 1:1 + len(features)] = features.T

            # shift elements according to offset for all observations
            phi = phi[np.arange(len(observation))[:, None], -offset[:, None] + np.arange(self.size())]

        return phi

    @abstractmethod
    def calc_features(self, observations):
        """
        calc the features according to the implemented method
        :param observations: vector of samples to compute the features for
        :return: feature vector for each sample
        """
        raise NotImplementedError

    @abstractmethod
    def size(self):
        """
        :return: total length of feature vector
        """
        raise NotImplementedError

    @abstractmethod
    def get_offset(self, actions):
        """
        get offset in phi for action or vector of actions.
        :param actions: actions the offset is required for
        :return: offset values
        """
        raise NotImplementedError
