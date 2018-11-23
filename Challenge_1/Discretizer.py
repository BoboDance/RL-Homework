import numpy as np


class Discretizer(object):

    def __init__(self, n_bins, space):
        self.space = space
        self.n_bins = n_bins

        self.high = self.space.high
        self.low = self.space.low

        self.bins = np.array([np.linspace(self.low[i] - 1e-5, self.high[i], self.n_bins, endpoint=False) for i in
                              range(self.space.shape[0])])

    def discretize(self, state):
        """
        bins the state
        :param state:
        :return: discretized state
        """
        s_dis = np.zeros(state.shape, dtype="int32")
        for i, s in enumerate(state):
            s_dis[i] = np.searchsorted(self.bins[i], s)
        return s_dis - 1
