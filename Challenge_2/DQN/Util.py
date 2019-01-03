import matplotlib.pyplot as plt
import numpy as np
import torch
import os

def create_initial_samples(env, memory, count, discrete_actions):
    last_observation = env.reset()
    samples = 0
    while samples < count:
        action_idx = np.random.choice(range(len(discrete_actions)), 1)
        action = discrete_actions[action_idx]
        # np.clip(np.random.normal(action, 1), env.action_space.low, env.action_space.high)
        observation, reward, done, _ = env.step(action)
        memory.push((*last_observation, *action_idx, reward, *observation, done))
        samples += 1

        if done:
            last_observation = env.reset()


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
