import numpy as np

from Challenge_2.LSPI.BasisFunctions.BasisFunction import BasisFunction


class Policy:

    def __init__(self, basis_function: BasisFunction, n_actions: int, weights: np.ndarray = None, eps: float = 1,
                 tie_breaker: str = "first"):
        """

        :param basis_function: Basisfunction object which returns phi upon call
        :param n_actions: ndarray of actions
        :param weights: inital weights, if not provided sample uniform weights
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

    def Q(self, state: np.ndarray, action_idx: np.ndarray = None) -> np.ndarray:
        """
        returns Q value for state action pair. If action is none value for all actions is returned.
        :param state: ndarray [batch_size x state_dim]
        :param action_idx: ndarray [batch_size x action_dim]
        :return:
        """
        phi = self.basis_function(state, action_idx)
        return phi @ self.w

    def calc_q_value(self, state, action):
        return self.w.dot(self.basis_function.evaluate(state, action))

    def best_action(self, state):
        q_values = [self.calc_q_value(state, action)
                    for action in range(self.n_actions)]

        best_q = float('-inf')
        best_actions = []

        for action, q_value in enumerate(q_values):
            if q_value > best_q:
                best_actions = [action]
                best_q = q_value
            elif q_value == best_q:
                best_actions.append(action)

        return best_actions[0]

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

    def choose_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Choose an action to take, either take action with highest Q or select random action with prob eps
        :param observation: ndarray [batch_size x state_dim]
        :return: action index
        """
        if np.random.uniform() <= self.eps:
            action_idx = np.random.choice(range(self.n_actions))
        else:
            action_idx = self.get_best_action(observation)

        return np.atleast_1d(action_idx)
