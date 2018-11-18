class Discretizer(object):

    def __init__(self, bins):
        self.bins = bins

    def discretize(self, state):
        raise NotImplementedError
