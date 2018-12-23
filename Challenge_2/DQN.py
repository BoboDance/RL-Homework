from typing import NamedTuple

import gym
import quanser_robots
import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
import torch.nn as nn

from Challenge_2.DQNModel import DQNModel
from Challenge_2.ReplayMemory import ReplayMemory
from Challenge_2.Util import create_initial_samples, DQNStatsFigure

env = gym.make("CartpoleSwingShort-v0")
np.random.seed(1)
env.seed(1)

dim_obs = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]

discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 5)
print("Used discrete actions: ", discrete_actions)

# transition: observation, action, reward, next observation, done
transition_size = dim_obs + dim_action + 1 + dim_obs + 1
REWARD_INDEX = dim_obs + dim_action
NEXT_OBS_INDEX = dim_obs + dim_action + 1
DONE_INDEX = -1

def get_y(transitions, discount_factor, old_model):
    y = np.empty((len(transitions), 1))

    for i, replay_entry in enumerate(transitions):
        if replay_entry[DONE_INDEX]:
            # if the episode is done after this step, only use the direct reward
            y[i] = replay_entry[REWARD_INDEX]
        else:
            # otherwise look ahead one step
            next_obs = replay_entry[NEXT_OBS_INDEX:NEXT_OBS_INDEX + dim_obs]
            y[i] = replay_entry[REWARD_INDEX] + discount_factor * old_model.get_best_value(next_obs, discrete_actions)

    return y

# the probability to choose an random action
epsilon = 0.01
discount_factor = 0.99

memory = ReplayMemory(5000, transition_size)
# The amount of random samples gathered before the learning starts (should be <= capacity of replay memory)
create_initial_samples(env, memory, 100)
# memory.plot_observations_cartpole()

# the size of the sampled minibatch used for learning in each step
minibatch_size = 7
# how many training episodes to do
training_episodes = 100

# try to predict the value of this state
model = DQNModel(dim_obs + dim_action, 1)
target_model = copy.deepcopy(model)
target_model.copy_state_dict(model)

# the amount of steps after which the old model is updated
old_model_update_steps = 10

# the maximum number of steps per episode
max_steps_per_episode = 500

reward_list = []
loss_list = []
mean_state_action_values_list = []

episode_state_action_values_list = []
episodes = 0
total_steps = 0
episode_steps = 0
episode_reward = 0
episode_loss = 0
detail_plot_episodes = 20
last_observation = env.reset()
stats_figure = DQNStatsFigure()
while episodes < training_episodes:
    # choose an action (random with prob. epsilon, otherwise choose the action with the best estimated value)
    if np.random.rand() <= epsilon:
        action = env.action_space.sample()
    else:
        action, value = model.get_best_action_and_value(last_observation, discrete_actions)
        episode_state_action_values_list.append(value)
        if episode_steps == 0:
            print("First step best value: {}".format(value))

    observation, reward, done, info = env.step(action)
    episode_reward += reward

    # save the observed step in our replay memory
    memory.push((*last_observation, *action, reward, *observation, done))

    # remember our last observation
    last_observation = observation

    # env.render()

    # learning step
    minibatch = memory.sample(minibatch_size)
    # calculate our y based on the old model
    y = get_y(minibatch, discount_factor, target_model)
    # perform gradient descend regarding to y on the current model
    episode_loss += model.gradient_descend_step(minibatch[:, 0:dim_obs + 1], y, loss_function=nn.MSELoss())

    total_steps += 1
    episode_steps += 1

    # copy the current model each old_model_update_steps steps
    if total_steps % old_model_update_steps == 0:
        target_model.copy_state_dict(model)

    if done or episode_steps >= max_steps_per_episode:
        last_observation = env.reset()
        episodes += 1

        state_action_values_mean = np.array(episode_state_action_values_list).mean()
        episode_state_action_values_list = []
        mean_state_action_values_list.append(state_action_values_mean)

        print("Episode {} > avg reward: {}, steps: {}, reward: {}, training loss: {}, avg value:"
              .format(episodes, episode_reward / episode_steps, episode_steps, episode_reward, episode_loss, state_action_values_mean))

        reward_list.append(episode_reward / episode_steps)
        loss_list.append(episode_loss)
        episode_steps = 0
        episode_loss = 0
        episode_reward = 0

        # update the plot
        stats_figure.draw(reward_list, mean_state_action_values_list, loss_list, episodes, detail_plot_episodes)


plt.show()
# memory.plot_observations_cartpole()

env.close()