import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from Challenge_2.Common.ReplayMemory import ReplayMemory

def plot_observations_cartpole(replay_memory: ReplayMemory):
    fig, ax = plt.subplots(3, 2)

    ax[0, 0].hist(replay_memory.memory[0:replay_memory.valid_entries, 0])
    ax[0, 0].set_title("x")
    ax[0, 1].hist(replay_memory.memory[0:replay_memory.valid_entries, 3])
    ax[0, 1].set_title("x_dot")

    ax[1, 0].hist(replay_memory.memory[0:replay_memory.valid_entries, 1])
    ax[1, 0].set_title("sin(theta)")
    ax[1, 1].hist(replay_memory.memory[0:replay_memory.valid_entries, 2])
    ax[1, 1].set_title("cos(theta)")
    ax[2, 0].hist(replay_memory.memory[0:replay_memory.valid_entries, 4])
    ax[2, 0].set_title("theta_dot")

    ax[2, 1].hist(replay_memory.memory[0:replay_memory.valid_entries, 5])
    ax[2, 1].set_title("action")

    fig.tight_layout()

    plt.show()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_saved_model(model, path, T, global_reward, optimizer=None):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        T.value = checkpoint['epoch']
        global_reward.value = checkpoint['global_reward']
        print("=> loaded checkpoint '{}' (T: {} -- global reward: {})"
              .format(path, checkpoint['epoch'], checkpoint['global_reward']))
    else:
        print("=> no checkpoint found at '{}'".format(path))


def get_best_action(Q, observation):
    # choose the action which gets the best value
    return [Q(np.atleast_2d(observation)).argmax(1).item()]


def get_best_values(Q, observations):
    return Q(np.atleast_2d(observations)).max(1)[0].detach().numpy()


def get_best_action_and_value(Q, observation, discrete_actions):
    values = Q(np.atleast_2d(observation))
    return np.array([discrete_actions[values.argmax(1).item()]]), values.max(1)[0].item()


class DQNStatsFigure():
    def __init__(self):
        self.fig, self.ax = plt.subplots(3, 2)

    def draw(self, reward_list, mean_state_action_values_list, loss_list, episodes, detail_plot_episodes):
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
