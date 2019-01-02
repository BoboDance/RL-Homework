import numpy as np


def calc_rbf(state, mean, beta):
    diff = (state - mean) ** 2
    return np.exp(-beta * np.sum(diff, axis=1))


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
        phi = np.zeros((len(action_idx), self.size(),))

        offset = self.means.shape[0] * action_idx.astype(np.int32)
        rbf = np.array([calc_rbf(state, mean, self.beta) for mean in self.means])

        # print np.sum(rbf,axis=0),1/(np.sum(rbf,axis=0))
        phi[np.arange(0, len(offset)), offset] = 1.
        offset_select = np.zeros((len(offset), len(rbf)))

        # TODO: Probably make it in matrix vector form
        for i, rbf_row in enumerate(rbf.T):
            phi[i, offset[i] + 1: offset[i] + 1 + len(rbf)] = rbf_row

        return phi

    def size(self):
        return (len(self.means) + 1) * self.n_actions
