import numpy as np


class Discretizer(object):

    def __init__(self, n_bins, space):
        self.space = space
        self.n_bins = n_bins

        self.high = self.space.high
        self.low = self.space.low

        self.bin_scaling = 1.025

        # TODO
        dense_locations = ["edge", "center", "center"]
        self.bins = np.array([self.increasing_bins(i, dense_locations[i]) for i in range(self.space.shape[0])])

        # self.bins = np.array([np.linspace(self.low[i] - 1e-10, self.high[i], self.n_bins + 1) for i in
        #                       range(self.space.shape[0])])

    def discretize(self, value):
        """
        :param value:
        :return: discretized state
        """
        s_dis = np.zeros(value.shape, dtype=np.int32)
        for i in range(value.shape[1]):
            s_dis[:, i] = np.searchsorted(self.bins[i][:-1], value[:, i])
        return s_dis - 1

    def scale_values_v2(self, value):
        # scale states to stay within action space
        total_range = self.high - self.low
        state_01 = value / (self.n_bins - 1)
        return (state_01 * total_range) + self.low

    def scale_values(self, value):
        # compute mean of lower and upper bound of bin
        scaled = np.zeros(value.shape, dtype=np.float32)
        for i in range(value.shape[1]):
            v = np.atleast_2d(value[:, i].T)
            scaled[:, i] = (self.bins[i][tuple(v)] + self.bins[i][tuple(v + 1)]) / 2
        return scaled

    def increasing_bins(self, dim, dense_location="center"):

        if dense_location not in ["center", "edge", "end", "start"]:
            raise ValueError("Invalid location of density found.")

        total_range = self.high[dim] - self.low[dim]

        if dense_location in ["center", "edge"]:

            bin_sizes = np.zeros(self.n_bins // 2)
            bin_sizes[0] = 1

            # scale bin based on previous one
            for i in range(1, self.n_bins // 2):
                bin_sizes[i] = bin_sizes[i - 1] * self.bin_scaling

            # normalize and scale to range
            bin_sizes /= np.sum(bin_sizes)
            bin_sizes *= total_range / 2

            if dense_location == "center":
                bin_sizes = np.concatenate([np.flip(bin_sizes), bin_sizes])
            elif dense_location == "edge":
                bin_sizes = np.concatenate([bin_sizes, np.flip(bin_sizes)])

        elif dense_location in ["end", "start"]:

            bin_sizes = np.zeros(self.n_bins - 1)
            bin_sizes[0] = 1

            for i in range(self.n_bins - 1):
                bin_sizes[i] = bin_sizes[i - 1] * self.bin_scaling

            # normalize and scale to range
            bin_sizes /= np.sum(bin_sizes)
            bin_sizes *= total_range

            if dense_location == "start":
                bin_sizes = np.flip(bin_sizes)

        bins = np.zeros(bin_sizes.shape[0])
        bins[0] = self.low[dim]

        for i in range(1, bins.shape[0]):
            bins[i] = bins[i - 1] + bin_sizes[i]

        # avoid having the exact min value --> bin would be -1 with self.discretize
        bins[0] -= 1e-10
        bins = np.append(bins, np.array([self.high[dim]]))

        return bins
