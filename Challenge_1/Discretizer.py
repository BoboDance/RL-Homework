import numpy as np


class Discretizer(object):

    def __init__(self, n_bins, space):
        self.space = space
        self.n_bins = n_bins

        self.high = self.space.high
        self.low = self.space.low

        self.bins = np.array([np.linspace(self.low[i] - 1e-8, self.high[i] + 1e-10, self.n_bins + 1) for i in
                              range(self.space.shape[0])])

    def discretize(self, state):
        """
        bins the state
        :param state:
        :return: discretized state
        """
        s_dis = np.empty(state.shape, dtype="int32")
        for i, s in enumerate(state):
            s_dis[i] = np.searchsorted(self.bins[i], s)
        return s_dis - 1
