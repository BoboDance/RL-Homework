"""
Evolutionary Strategies module for PyTorch models -- modified from https://github.com/staturecrane/PyTorch-ES
"""
import copy
from multiprocessing.pool import ThreadPool
import pickle

import numpy as np
import torch
import pprint


class NES(object):
    def __init__(self, weights, reward_func, population_size=50, sigma=0.1, learning_rate=0.001, decay=1.0,
                 sigma_decay=1.0, threadcount=4, render_test=False, reward_goal=None, consecutive_goal_stopping=None,
                 normalize_reward=True, save_path=None):
        """
        Initializes the learner.

        :param weights: The initial weights of the model
        :param reward_func: A partial function which can map weights to some reward measure
        :param population_size: The size of the population (random permutations per iteration)
        :param sigma: Parameter sigma which defines the strength of the jitter and
        :param learning_rate: The learning rate (defines step size)
        :param decay: Learning rate decay for each iteration
        :param sigma_decay: Sigma decay for each iteration
        :param threadcount: Number of threads used for learning
        :param render_test: Whether to render the reward_func
        :param reward_goal: Some goal reward used for early stopping
        :param consecutive_goal_stopping: The amount of times the reward_goal must be passed to stop training
        :param normalize_reward: Whether to normalize the rewards
        :param save_path: The path where the best weights will be saved
        """
        self.pop_size = population_size
        self.sigma = sigma
        self.lr = learning_rate

        # lr and exploration decay
        self.lr_decay = decay
        self.sigma_decay = sigma_decay

        # rendering
        self.render_test = render_test

        # early stopping
        self.reward_goal = reward_goal
        self.consecutive_goal_stopping = consecutive_goal_stopping
        self.consecutive_goal_count = 0

        # normalize reward
        self.normalize_reward = normalize_reward

        print("Parameter Settings:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.__dict__)

        self.pool = ThreadPool(threadcount)
        self.weights = weights
        self.reward_function = reward_func

        self.save_path = save_path

    def mutate_weights(self, weights, population: list = [], no_mutation: bool = False):
        """
        Add some random jitter to the given weights.

        :param weights: current set of weights
        :param population: population which defines mutation base
        :param no_mutation: add jitter or not, if false just get back weight vector
        :return: the mutated weights
        """
        new_weights = []
        for i, param in enumerate(weights):
            if no_mutation:
                new_weights.append(param.data)
            else:
                jittered = torch.from_numpy(self.sigma * population[i]).float()
                new_weights.append(param.data + jittered)

        return new_weights

    def run(self, iterations, print_mod=10):
        """
        Run NES for "iterations" iterations and evaluate the current weights after "print_mod" steps

        :param iterations: Number of iterations
        :param print_mod: Print frequency
        :return: The last weights
        """

        best_reward = -np.inf

        for iteration in range(iterations):

            # init pop randomly, which specifies change in params
            population = []
            for _ in range(self.pop_size):
                x = []
                for param in self.weights:
                    x.append(np.random.randn(*param.data.size()))
                population.append(x)

            # compute rewards/fitness for pop
            rewards = self.pool.map(self.reward_function,
                                    [self.mutate_weights(copy.deepcopy(self.weights), population=pop) for pop in
                                     population])
            if np.std(rewards) != 0:

                if self.normalize_reward:
                    rewards = (rewards - np.mean(rewards)) / np.std(rewards)

                # weight update via natural gradient descent
                for index, param in enumerate(self.weights):
                    A = np.array([p[index] for p in population])
                    rewards_pop = torch.from_numpy(np.dot(A.T, rewards).T).float()
                    param.data = param.data + self.lr / (self.pop_size * self.sigma) * rewards_pop

                    # lr decay and sigma decay
                    self.lr *= self.lr_decay
                    self.sigma *= self.sigma_decay

            # compute reward for test run
            if (iteration + 1) % print_mod == 0:
                test_reward_total = 0
                steps_total = 0
                for _ in range(10):
                    test_reward, steps = self.reward_function(
                        self.mutate_weights(copy.deepcopy(self.weights), no_mutation=True), render=self.render_test,
                        return_steps=True)
                    test_reward_total += test_reward
                    steps_total += steps

                test_reward_total /= 10
                steps_total /= 10

                print(
                    f'Iteration: {iteration + 1:d} -- reward: {test_reward_total:f}, steps: {steps_total} '
                    f'-- sigma: {self.sigma:.7f} -- lr: {self.lr:.7}')

                # save model weights
                if test_reward_total > best_reward:
                    if self.save_path is not None:
                        pickle.dump(self.weights, open(self.save_path, 'wb'))
                        print(f"Best model with {test_reward_total} saved.")
                    best_reward = test_reward_total

                # early stopping if threshold is crossed consecutive_goal_stopping times
                if self.reward_goal and self.consecutive_goal_stopping:
                    if test_reward_total >= self.reward_goal:
                        self.consecutive_goal_count += 1
                    else:
                        self.consecutive_goal_count = 0

                    if self.consecutive_goal_count >= self.consecutive_goal_stopping:
                        return self.weights

        return self.weights
