import numpy as np


class Policy:

    def __init__(self, basis_function, n_actions, weights=None, eps=1, tie_breaker="first"):
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
        self.eps = eps

        if weights is None:
            self.w = np.random.uniform(-1.0, 1.0, size=(basis_function.size(),))
        else:
            self.w = weights

    def Q(self, state, action_idx=None):
        """
        returns Q value for state action pair. If action is none value for all actions is returned.
        :param state: ndarray [batch_size x state_dim]
        :param action_idx: ndarray [batch_size x action_dim]
        :return:
        """
        phi = self.basis_function(state, action_idx)
        return phi @ self.w

    def get_best_action(self, observation):

        observation = np.atleast_2d(observation)

        # action_idx is None, therefore all actions are evaluated
        Q_values = self.Q(observation)

        if self.tie_breaker == "first":
            return np.argmax(Q_values, axis=1)

        elif self.tie_breaker == "last":
            return np.argmax(reversed(Q_values), axis=1)

        elif self.tie_breaker == "random":
            best_actions = np.argwhere(Q_values == np.max(Q_values, axis=1))
            return np.random.choice(best_actions)

    def choose_action(self, observation):
        if np.random.uniform() <= self.eps:
            action_idx = np.random.choice(range(self.n_actions))
        else:
            action_idx = self.get_best_action(observation)

        return np.atleast_1d(action_idx)
