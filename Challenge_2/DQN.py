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
from Challenge_2.Util import create_initial_samples

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
rewards = []
episodes = 0
total_steps = 0
episode_steps = 0
episode_reward = 0
episode_loss = 0
fig, ax = plt.subplots(2, 1)
last_observation = env.reset()
while episodes < training_episodes:
    # choose an action (random with prob. epsilon, otherwise choose the action with the best estimated value)
    if np.random.rand() <= epsilon:
        action = env.action_space.sample()
    else:
        action = model.get_best_action(last_observation, discrete_actions)
        if episode_steps == 0:
            print("First step best value: {}".format(model.get_best_value(last_observation, discrete_actions)))

    observation, reward, done, info = env.step(action)
    episode_reward += reward
    rewards.append(reward)

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
        print("Episode {} > avg reward: {}, steps: {}, reward: {}, training loss: {}".format(episodes, episode_reward / episode_steps, episode_steps, episode_reward, episode_loss))
        reward_list.append(episode_reward / episode_steps)
        loss_list.append(episode_loss)
        episode_steps = 0
        episode_loss = 0
        episode_reward = 0

        # update the plot
        ax[0].cla()
        ax[1].cla()
        ax[0].plot(reward_list, c="blue")
        ax[0].set_title("Episode Average Reward")
        ax[1].plot(loss_list, c="red")
        ax[1].set_title("Episode Total Loss")
        fig.tight_layout()

        plt.draw()
        plt.pause(0.001)

plt.show()
plt.hist(rewards)
plt.show()
# memory.plot_observations_cartpole()

env.close()