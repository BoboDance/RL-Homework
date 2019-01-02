import numpy as np


def calc_rbf(state, mean, beta):
    diff = (state - mean) ** 2
    return np.exp(-beta * np.sum(diff))


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

    def __call__(self, state, action_idx):
        phi = np.zeros((self.size(),))
        offset = (len(self.means[0]) + 1) * int(action_idx)

        rbf = [calc_rbf(state, mean, self.beta) for mean in self.means]

        # print np.sum(rbf,axis=0),1/(np.sum(rbf,axis=0))
        phi[offset] = 1.
        phi[offset + 1: offset + 1 + len(rbf)] = rbf

        return phi

    def size(self):
        return (len(self.means) + 1) * self.n_actions
