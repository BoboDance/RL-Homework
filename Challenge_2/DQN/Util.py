import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from Challenge_2.Common.ReplayMemory import ReplayMemory
from Challenge_2.Common.Util import normalize_state
from Challenge_2.DQN.Models import DQNModel
import gym


def get_current_lr(optimizer) -> float:
    """
    Return the current learning rate
    :param optimizer: Optimizer object
    :return:
    """
    return optimizer.param_groups[0]['lr']


def save_model(env: gym.Env, Q:DQNModel, episodes: int, reward: float):
    """
    Saves the current DQNModel in the checkpoint directory
    :param env: Gym environment handle
    :param Q: Q-Model handle
    :param episodes: Number of episodes trained
    :param reward: Achieved rewards for the given model
    :return:
    """
    torch.save({
        'env_id': env.unwrapped.spec.id,
        'episodes': episodes,
        'Q': Q.state_dict(),
        'reward': reward,
    }, "./checkpoints/Q_{}_{}_{:.2f}.pth.tar".format(env.unwrapped.spec.id, episodes, reward))


def load_model(env, Q: DQNModel, path: str):
    """
    Loads the weights of a model given a the path.
    :param env: Gym environment handle
    :param Q: Architecture description of the model
    :param path: Absolute path to the weights including the filename
    :return:
    """
    if os.path.isfile(path):
        checkpoint = torch.load(path)

        if checkpoint['env_id'] != env.unwrapped.spec.id:
            print("The model you are trying to load was created for a different environment!")
            return

        Q.load_state_dict(checkpoint['Q'])

        print("Loaded model from '{}'".format(path))
    else:
        print("No model found at '{}'".format(path))


def get_best_action(Q: DQNModel, observation: np.ndarray) -> [int]:
    """
    Choose the action which gets the best value
    :param Q: Handle of the DQN model
    :param observation: Current environment state
    :return: List of the index to the bin of the best action
    """
    return [Q(np.atleast_2d(observation)).argmax(1).item()]


def get_best_values(Q: DQNModel, observations: np.ndarray) -> np.ndarray:
    """
    Return the best possible future values given an observation
    :param Q: Handle of the DQN model
    :param observations: Numpy array of the env-state
    :return: Numpy matrix of the best values
    """
    return Q(np.atleast_2d(observations)).max(1)[0].detach().numpy()


def get_best_action_and_value(Q: DQNModel, observation: np.ndarray, discrete_actions: np.ndarray) -> [np.ndarray, np.ndarray]:
    """
    Returns a tuple of the best action and its corresponding value
    :param Q: Handle of the Q-Model
    :param observation: Numpy array of the env-state
    :param discrete_actions: Numpy storting the corresponding actions for the bins
    :return: Returns a tuple of the best action and its corresponding value
    """
    values = Q(np.atleast_2d(observation))
    return np.array([discrete_actions[values.argmax(1).item()]]), values.max(1)[0].item()


def get_policy_fun(env, Q: DQNModel, normalize: bool, low: np.ndarray = None, high: np.ndarray = None) -> callable:
    """
    Returns function handle for the learnt policy
    :param env: Handel fo the gym environment
    :param Q: Handle of the DQN model
    :param normalize: Boolean indicating if the environment state space shall be normalized
    :param low: Numpy array describing the lowest numerical values for the env observation
    :param high: Numpy array describing the highest numerical values for the env observation
    :return:
    """
    def policy(obs):

        if normalize:
            obs = normalize_state(env, obs, low=low, high=high)

        Q.eval()
        action_idx = get_best_action(Q, obs)
        return Q.discrete_actions[action_idx]

    return policy


class CartpoleReplayMemoryFigure(object):
    def __init__(self, replay_memory: ReplayMemory, discrete_actions: np.ndarray):
        """
        Constructor
        :param replay_memory: Handle to the replay memory
        :param discrete_actions: Numpy array of the choosen actions
        """
        self.fig, self.ax = plt.subplots(3, 2)
        self.replay_memory = replay_memory
        self.discrete_actions = discrete_actions

    def draw(self):
        """
        Plots histograms for the taken actions and the environment states
        :return:
        """
        self.ax[0, 0].cla()
        self.ax[0, 1].cla()
        self.ax[1, 0].cla()
        self.ax[1, 1].cla()
        self.ax[2, 0].cla()
        self.ax[2, 1].cla()

        self.ax[0, 0].hist(self.replay_memory.memory[0:self.replay_memory.valid_entries, 0])
        self.ax[0, 0].set_title("x")
        self.ax[0, 1].hist(self.replay_memory.memory[0:self.replay_memory.valid_entries, 3])
        self.ax[0, 1].set_title("x_dot")

        self.ax[1, 0].hist(self.replay_memory.memory[0:self.replay_memory.valid_entries, 1])
        self.ax[1, 0].set_title("sin(theta)")
        self.ax[1, 1].hist(self.replay_memory.memory[0:self.replay_memory.valid_entries, 2])
        self.ax[1, 1].set_title("cos(theta)")
        self.ax[2, 0].hist(self.replay_memory.memory[0:self.replay_memory.valid_entries, 4])
        self.ax[2, 0].set_title("theta_dot")

        action_indices = self.replay_memory.memory[0:self.replay_memory.valid_entries, 5].astype(int)
        self.ax[2, 1].hist(self.discrete_actions[action_indices], bins=(len(self.discrete_actions) * 2 - 1))
        self.ax[2, 1].set_title("action")

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.001)


class DQNStatsFigure(object):

    def __init__(self):
        """
        Constructor
        """
        self.fig, self.ax = plt.subplots(3, 2)

    def draw(self, reward_list: list, mean_state_action_values_list: list, loss_list: list, episodes: list,
             detail_plot_episodes: bool):
        """
        Plots the metrics during training using matplotlib
        :param reward_list: List which stored all passed rewards
        :param mean_state_action_values_list: List of the taken action and their corresponding values
        :param loss_list: List of all the losses
        :param episodes: List of the trained
        :param detail_plot_episodes: Boolean defining if a detail plot shall be created
        :return:
        """
        self.ax[0, 0].cla()
        self.ax[0, 0].plot(reward_list, c="blue")
        self.ax[0, 0].set_title("Average Reward")
        self.ax[1, 0].cla()
        self.ax[1, 0].plot(mean_state_action_values_list, c="orange")
        self.ax[1, 0].set_title("Average Value")
        self.ax[2, 0].cla()
        self.ax[2, 0].plot(loss_list, c="red")
        self.ax[2, 0].set_title("Total Loss")

        if episodes >= detail_plot_episodes:
            detail_x = np.arange(episodes - detail_plot_episodes, episodes)
        else:
            detail_x = np.arange(0, episodes)

        self.ax[0, 1].cla()
        self.ax[0, 1].plot(detail_x, reward_list[-detail_plot_episodes:], c="blue")
        self.ax[0, 1].set_title("Average Reward")
        self.ax[1, 1].cla()
        self.ax[1, 1].plot(detail_x, mean_state_action_values_list[-detail_plot_episodes:], c="orange")
        self.ax[1, 1].set_title("Average Value")
        self.ax[2, 1].cla()
        self.ax[2, 1].plot(detail_x, loss_list[-detail_plot_episodes:], c="red")
        self.ax[2, 1].set_title("Total Loss")

        self.fig.tight_layout()
        plt.draw()
        plt.pause(0.001)

