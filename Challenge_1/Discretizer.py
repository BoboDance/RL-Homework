import numpy as np
from sklearn.preprocessing import KBinsDiscretizer


class Discretizer(object):

    def __init__(self, bins, space):
        self.observation_space = space
        self.bins = bins
        self.discretizer = KBinsDiscretizer(n_bins=self.bins, encode='ordinal')

        high = self.observation_space.high
        low = self.observation_space.low
        self.discretizer.fit(np.concatenate([high, low], axis=0).reshape(-1, self.observation_space.shape[0]))

    def discretize(self, state):
        np.array(state)
        return self.discretizer.transform(np.atleast_2d(state))
