import math

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tensorboardX import SummaryWriter

from Challenge_2.DQN.DQNModel import DQNModel
from Challenge_2.DQN.ReplayMemory import ReplayMemory
from Challenge_2.DQN.Util import create_initial_samples, save_checkpoint, get_best_action, get_best_value

import quanser_robots

seed = 1

torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make("CartpoleSwingShort-v0")
env.seed(seed)

dim_obs = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]

discrete_actions = np.linspace(-5, 5, 20)
# discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 20)
print("Used discrete actions: ", discrete_actions)

# transition: observation, action, reward, next observation, done
transition_size = dim_obs + dim_action + 1 + dim_obs + 1
ACTION_INDEX = dim_obs
REWARD_INDEX = dim_obs + dim_action
NEXT_OBS_INDEX = dim_obs + dim_action + 1
DONE_INDEX = -1

# Enable if double Q learning is allowed for challenge
use_double_Q = False

# the probability to choose an random action decaying over time
eps_start = .5
eps_end = 0.05
eps_decay = 200

gamma = 0.99


def get_expected_values(transitions, model):
    global REWARD_INDEX
    global DONE_INDEX
    global NEXT_OBS_INDEX
    global dim_obs
    global gamma

    y = np.zeros((len(transitions), 1))

    # TODO to make this in one step and do not use loop
    for i, replay_entry in enumerate(transitions):
        y[i] = replay_entry[REWARD_INDEX]
        if not replay_entry[DONE_INDEX]:
            # if non terminal state is present look ahead one step
            next_obs = replay_entry[NEXT_OBS_INDEX:NEXT_OBS_INDEX + dim_obs]
            y[i] += gamma * get_best_value(model, next_obs)

    return y


def choose_action(Q, observation, discrete_actions):
    global total_steps
    # compute decaying eps_threshold to encourage exploration at the beginning
    eps_threshold = eps_end + (eps_start - eps_end) * math.exp(-1. * total_steps / eps_decay)
    # choose epsilon-greedy action
    if np.random.uniform() <= eps_threshold:
        action_idx = np.random.choice(range(len(discrete_actions)), 1)
    else:
        action_idx = get_best_action(Q, observation)

    return action_idx


def optimize(memory, Q, target_Q, use_double_Q=False, criterion=nn.SmoothL1Loss()):
    global minibatch_size
    global gamma
    global dim_obs
    global NEXT_OBS_INDEX

    Q.train()

    minibatch = memory.sample(minibatch_size)

    # calculate our y based on the target model
    expected_values = get_expected_values(minibatch, target_Q)
    expected_values = torch.from_numpy(expected_values).float()

    obs = minibatch[:, 0:dim_obs]
    # obs = torch.from_numpy(obs).float()

    actions = minibatch[:, ACTION_INDEX:REWARD_INDEX]
    actions = torch.from_numpy(actions).long()

    # perform gradient descend regarding to expected values on the current model
    Q.optimizer.zero_grad()

    values = Q(obs).gather(1, actions)
    loss = criterion(values, expected_values)
    loss.backward()

    torch.nn.utils.clip_grad_norm_(Q.parameters(), 1)

    Q.optimizer.step()

    return loss.item()


memory = ReplayMemory(5000, transition_size)
# The amount of random samples gathered before the learning starts (should be <= capacity of replay memory)
create_initial_samples(env, memory, 500, discrete_actions)
# memory.plot_observations_cartpole()

# the size of the sampled minibatch used for learning in each step
minibatch_size = 128
# how many training episodes to do
max_episodes = 5000

# init tensorboard writer for better visualizations
writer = SummaryWriter()

# init NNs for Q function approximation
Q = DQNModel(n_inputs=dim_obs, n_outputs=len(discrete_actions), optimizer="rmsprop")
target_Q = DQNModel(n_inputs=dim_obs, n_outputs=len(discrete_actions), optimizer=None)
target_Q.load_state_dict(Q.state_dict())

# the amount of steps after which the target model is updated
target_model_update_steps = 10

# the maximum number of steps per episode
max_steps_per_episode = 2500

# reward_list = []
# loss_list = []
# mean_value_list = []

episodes = 0
total_steps = 0

episode_steps = 0
episode_reward = 0
episode_loss = 0

# stats_figure = DQNStatsFigure()
# detail_plot_episodes = 20
obs = env.reset()

while episodes < max_episodes:

    action_idx = choose_action(Q, obs, discrete_actions)
    action = discrete_actions[action_idx]

    next_obs, reward, done, _ = env.step(action)
    # reward clipping
    reward = min(max(-1., reward), 1.)
    episode_reward += reward

    # save the observed step in our replay memory
    memory.push((*obs, *action_idx, reward, *next_obs, done))

    # remember last observation
    obs = next_obs

    env.render()

    loss = optimize(memory, Q, target_Q, criterion=nn.MSELoss())
    episode_loss += loss

    writer.add_scalar("loss", loss, total_steps)

    total_steps += 1
    episode_steps += 1

    # copy the current model each target_model_update_steps steps
    if total_steps % target_model_update_steps == 0:
        target_Q.load_state_dict(Q.state_dict())
        # target_model.copy_state_dict(model)

    if done or episode_steps >= max_steps_per_episode:
        obs = env.reset()
        episodes += 1

        # value_mean = np.array(episode_value_list).mean()
        # episode_value_list = []
        # mean_value_list.append(value_mean)

        avg_reward = episode_reward / episode_steps

        # save model after each epoch
        save_checkpoint({
            'epoch': total_steps,
            'episodes': episodes,
            'Q': Q.state_dict(),
            'target_Q': target_Q.state_dict(),
            'reward': episode_reward,
            'optimizer': Q.optimizer.state_dict(),
        }, filename="./checkpoints/Q_model_epoch-{}_reward-{}.pth.tar".format(total_steps, episode_reward))

        print("Episode {:5d} -- total steps: {:8d} > avg reward: {:.10f} -- steps: {:4d} -- reward: {:5.5f} "
              "-- training loss: {:10.5f}"
              .format(episodes, total_steps, avg_reward, episode_steps, episode_reward, episode_loss))

        # writer.add_scalar("avg_value", value_mean, episodes)
        writer.add_scalar("avg_reward", avg_reward, episodes)
        writer.add_scalar("total_reward", episode_reward, episodes)

        # reward_list.append(episode_reward / episode_steps)
        # loss_list.append(episode_loss)

        episode_steps = 0
        episode_loss = 0
        episode_reward = 0

        # update the plot
        # stats_figure.draw(reward_list, mean_value_list, loss_list, episodes, detail_plot_episodes)

plt.show()
# memory.plot_observations_cartpole()

env.close()
