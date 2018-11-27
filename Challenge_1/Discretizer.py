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
        for i in range(state.shape[1]):
            s_dis[:, i] = np.searchsorted(self.bins[i], state[:, i])
        return s_dis - 1

    def scale_values(self, value):
        # scale states to stay within action space
        total_range = (self.high - self.low)
        state_01 = value / (self.n_bins - 1)
        return (state_01 * total_range) + self.low
