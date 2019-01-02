import numpy as np


class Policy:

    def __init__(self, basis_function, n_actions, weights=None, gamma=.99, eps=1, tie_breaker="first"):
        """

        :param basis_function: Basisfunction object which returns phi upon call
        :param n_actions: ndarray of actions
        :param weights: inital weights, if not provided sample uniform weights
        :param gamma: discount factor
        :param eps: eps-greedy policy parameter
        :param tie_breaker:
        """

        self.basis_function = basis_function
        self.tie_breaker = tie_breaker

        self.n_actions = n_actions
        self.gamma = gamma
        self.eps = eps

        if weights is None:
            self.w = np.random.uniform(-1.0, 1.0, size=(basis_function.size(),))
        else:
            self.w = weights

    def Q(self, state, action):

        phi = self.basis_function(state, action)
        return phi.dot(self.w)

    def get_best_action(self, observation):

        observation = np.atleast_2d(observation)
        #Q_values = [self.Q(observation, action) for action in range(self.n_actions)]
        Q_values = np.array([self.Q(np.tile(obs, self.n_actions).reshape(self.n_actions, -1),
                           np.arange(self.n_actions)) for obs in observation])


        if self.tie_breaker == "first":
            return np.argmax(Q_values, axis=1)

        elif self.tie_breaker == "last":
            return np.argmax(reversed(Q_values), axis=1)

        elif self.tie_breaker == "random":
            best_actions = np.argwhere(Q_values == np.max(Q_values, axis=1))
            return np.random.choice(best_actions)

    def choose_action(self, observation):
        if np.random.uniform() <= self.eps:
            action = np.random.choice(range(self.n_actions))
        else:
            action = self.get_best_action(observation)

        return np.atleast_1d(action)
