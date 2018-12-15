import numpy as np


class Discretizer(object):

    def __init__(self, high: [int], low: [int], n_bins_per_feature: [int], dense_locations=None, bin_scaling=1.025):
        """

        :param n_bins_per_feature: Describes the number of bins per feature
        :param space: Feature space with .high and .low attribute (e.g. state_space or action_space)
        :param dense_locations: Possible options: [string] or None
                                - None: Equal sized bins across the full space
                                - "center": Higher density of bins at the center
                                - "edge": More bins at both edges
                                - "start": More bins at the start
                                - "end": More bins at the end regions
        """

        self.n_bins_per_feature = n_bins_per_feature
        self.bin_scaling = bin_scaling

        self.high = high
        self.low = low

        self.bins = []

        for i in range(len(n_bins_per_feature)):

            if dense_locations[i] == "equal":
                # we add +1 for the upper boundary.
                b = np.linspace(self.low[i] - 1e-10, self.high[i], self.n_bins_per_feature[i] + 1)
            else:
                b = self.increasing_bins(i, dense_locations[i])
            self.bins.append(b)

    def discretize(self, value):
        """
        :param value:
        :return: discretized state
        """
        s_dis = np.zeros(value.shape, dtype=np.int32)
        for i in range(value.shape[1]):
            s_dis[:, i] = np.searchsorted(self.bins[i][:-1], value[:, i])
        return s_dis - 1

    def scale_values(self, value):
        # compute mean of lower and upper bound of bin
        scaled = np.zeros(value.shape, dtype=np.float32)
        for i in range(value.shape[1]):
            v = np.atleast_2d(value[:, i].T)
            # calculate the mean from the left and right current bins
            scaled[:, i] = (self.bins[i][tuple(v)] + self.bins[i][tuple(v + 1)]) / 2
        return scaled

    def scale_values_stochastic(self, value, n_samples):
        # compute mean of lower and upper bound of bin
        scaled = np.zeros((value.shape[0], n_samples, value.shape[1]), dtype=np.float32)
        for i in range(value.shape[1]):
            v = np.atleast_2d(value[:, i].T)
            # idx = tuple(v)
            bins = self.bins[i]
            low = bins[tuple(v)]
            high = bins[tuple(v + 1)]

            # Gaussian MC sampling within the bin
            # sigma = (high - low) / 3
            # mu = (low + high) / 2
            # # 99.7% of samples are in 3*sigma range, clip others
            # scaled[:, :, i] = np.clip(np.random.normal(mu, sigma, size=(n_samples, len(mu))), low, high).T

            scaled[:, :, i] = np.random.uniform(low, high, size=(n_samples, len(low))).T

        return scaled

    def increasing_bins(self, dim, dense_location="center"):

        if dense_location not in ["center", "edge", "end", "start"]:
            raise ValueError("Invalid location of density.")

        total_range = self.high[dim] - self.low[dim]

        if dense_location in ["center", "edge"]:

            bin_sizes = np.zeros(self.n_bins_per_feature[dim] // 2)
            bin_sizes[0] = 1

            # scale bin based on previous one
            for i in range(1, self.n_bins_per_feature[dim] // 2):
                bin_sizes[i] = bin_sizes[i - 1] * self.bin_scaling

            # normalize and scale to range
            bin_sizes /= np.sum(bin_sizes)
            bin_sizes *= total_range / 2

            if dense_location == "center":
                bin_sizes = np.concatenate([np.flip(bin_sizes), bin_sizes])
            elif dense_location == "edge":
                bin_sizes = np.concatenate([bin_sizes, np.flip(bin_sizes)])

        elif dense_location in ["end", "start"]:

            bin_sizes = np.zeros(self.n_bins_per_feature[dim])
            bin_sizes[0] = 1

            for i in range(1, self.n_bins_per_feature[dim]):
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

        # convert the bins to a numpy array
        bins = np.array(bins)

        return bins
