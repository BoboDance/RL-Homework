import matplotlib.pyplot as plt
import numpy as np
import torch
import os

from Challenge_2.Common.ReplayMemory import ReplayMemory
from Challenge_2.DQN import DQNModel


def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def save_model(env, Q: DQNModel, episodes, reward):
    torch.save({
        'env_id': env.unwrapped.spec.id,
        'episodes': episodes,
        'Q': Q.state_dict(),
        'reward': reward,
    }, "./checkpoints/Q_{}_{}_{:.2f}.pth.tar".format(env.unwrapped.spec.id, episodes, reward))


def load_model(env, Q: DQNModel, path):
    if os.path.isfile(path):
        checkpoint = torch.load(path)

        if checkpoint['env_id'] != env.unwrapped.spec.id:
            print("The model you are trying to load was created for a different environment!")
            return

        Q.load_state_dict(checkpoint['Q'])

        print("Loaded model from '{}'".format(path))
    else:
        print("No model found at '{}'".format(path))


def get_best_action(Q, observation):
    # choose the action which gets the best value
    return [Q(np.atleast_2d(observation)).argmax(1).item()]


def get_best_values(Q, observations):
    return Q(np.atleast_2d(observations)).max(1)[0].detach().numpy()


def get_best_action_and_value(Q, observation, discrete_actions):
    values = Q(np.atleast_2d(observation))
    return np.array([discrete_actions[values.argmax(1).item()]]), values.max(1)[0].item()


def get_policy_fun(Q):
    def policy(obs):
        Q.eval()
        action_idx = get_best_action(Q, obs)
        return Q.discrete_actions[action_idx]

    return policy


class CartpoleReplayMemoryFigure():
    def __init__(self, replay_memory: ReplayMemory, discrete_actions):
        self.fig, self.ax = plt.subplots(3, 2)
        self.replay_memory = replay_memory
        self.discrete_actions = discrete_actions

    def draw(self):
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
