import matplotlib.pyplot as plt
import numpy as np

def create_initial_samples(env, memory, count):
    last_observation = env.reset()
    samples = 0
    action = env.action_space.sample()
    while samples < count:
        action = env.action_space.sample()  # np.clip(np.random.normal(action, 1), env.action_space.low, env.action_space.high)
        observation, reward, done, info = env.step(action)
        memory.push((*last_observation, *action, reward, *observation, done))
        samples += 1

        if done:
            last_observation = env.reset()
            action = env.action_space.sample()

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