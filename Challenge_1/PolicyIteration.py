import numpy as np

from Challenge_1 import Discretizer


class PolicyIteration(object):

    def __init__(self, env, dynamics_model, reward_model, discretizer_state: Discretizer,
                 discretizer_action: Discretizer, discount=1., theta=1e-9):

        self.env = env
        self.discretizer_state = discretizer_state
        self.discretizer_action = discretizer_action

        self.state_dim = self.discretizer_state.bins
        self.n_actions = self.discretizer_action.bins

        self.dynamics_model = dynamics_model
        self.reward_model = reward_model

        self.discount = discount
        self.theta = theta

        # Could also be a dict
        self.value_function = np.zeros((self.state_dim,))

        # Random policy with equally likely actions
        self.policy = np.ones([self.state_dim, self.n_actions]) / self.n_actions

    def _policy_evaluation(self, max_iter=1000000):

        for i in range(max_iter):

            delta = 0
            for state in range(self.state_dim):

                state_dis = self.discretizer_state.discretize(state)
                current_value = 0

                # Try out all possible actions for this state
                for action, prob in enumerate(self.policy[state_dis]):
                    # compute actions based on models without interaction with env
                    s_a = np.concatenate([state, action])
                    state_prime = self.dynamics_model.predict(s_a)
                    state_prime_dis = self.discretizer_state.discretize(state_prime)
                    reward = self.reward_model.predict(s_a)

                    # Calculate the expected value of next state
                    current_value += prob * (reward + self.discount * self.value_function[state_prime_dis])

                # Change of value function
                delta = np.maximum(delta, np.abs(self.value_function[state] - current_value))
                self.value_function[state] = current_value

            # Terminate if change is below threshold
            if delta < self.theta:
                print('Policy evaluation finished in {} iterations.'.format(i + 1))
                break

    def _get_action_dist(self, state):

        action_prob = np.zeros((self.n_actions,))

        for action in range(self.n_actions):
            # compute actions based on models without interaction with env
            s_a = np.concatenate([state, action])
            state_prime = self.dynamics_model.predict(s_a)
            state_prime_dis = self.dynamics_model.discretize(state_prime)
            reward = self.reward_model.predict(s_a)

            action_prob[action] += (reward + self.discount * self.value_function[state_prime_dis])

        # normalize to sum up to one
        action_prob /= np.sum(action_prob)
        return action_prob

    def run(self, max_iter=100000):

        for i in range(max_iter):
            stable = True

            # determines value function
            self._policy_evaluation(max_iter=max_iter)

            # policy improvement
            for state in range(self.state_dim):
                # Choose action with current policy
                policy_action = np.argmax(self.policy[state])

                # Check if current action is actually best
                action_prob = self._get_action_dist(state)
                best_action = np.argmax(action_prob)

                # If action didn't change
                if policy_action != best_action:
                    stable = True
                    # Greedy policy update
                    self.policy[state] = np.eye(self.n_actions)[best_action]

            # policy iteration converged
            if stable:
                print('Evaluated {} policies and found stable policy'.format(i + 1))
                break
