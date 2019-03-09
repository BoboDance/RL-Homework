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
                 seed=None, normalize_reward=True):

        np.random.seed(seed)
        self.pop_size = population_size
        self.sigma = sigma
        self.lr = learning_rate

        # lr and exploration decay
        self.decay = decay
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

    def mutate_weights(self, weights, population: list = [], no_mutation: bool = False):
        """
        Add some random jitter to params or  if
        :param weights: current set of weights
        :param population: population which defines mutation base
        :param no_mutation: add jitter or not, if false just get back weight vector
        :return: new_weights
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
        run NES for iterations and print after mod steps
        :param iterations: number of steps
        :param print_mod: print frequency
        :return:
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
                    self.lr *= self.decay
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
                    pickle.dump(self.weights, open(f"../checkpoints/reward-{test_reward_total}.pkl", 'wb'))
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
