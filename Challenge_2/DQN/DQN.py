import math

import quanser_robots
import gym
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from Challenge_2.Common.ReplayMemory import ReplayMemory
from Challenge_2.Common.Util import create_initial_samples
from Challenge_2.DQN.DQNModel import DQNModel
from Challenge_2.DQN.Util import get_best_values, get_best_action, get_current_lr

# init tensorboard writer for better visualizations
writer = SummaryWriter()

seed = 1

torch.manual_seed(seed)
np.random.seed(seed)

# env = gym.make("Pendulum-v0")
env = gym.make("CartpoleSwingShort-v0")
env.seed(seed)

dim_obs = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]

# discrete_actions = np.array([-2, 2])
discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 2)
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
eps_start = 1.
eps_end = 0.1
eps_decay = 2000

gamma = 0.999


def get_expected_values(transitions, model):
    global REWARD_INDEX
    global DONE_INDEX
    global NEXT_OBS_INDEX
    global dim_obs
    global gamma

    y = transitions[:, REWARD_INDEX].reshape(-1, 1)
    not_done_filter = ~transitions[:, DONE_INDEX].astype(bool)
    next_obs = transitions[not_done_filter, NEXT_OBS_INDEX:NEXT_OBS_INDEX + dim_obs]
    y[not_done_filter] += gamma * get_best_values(model, next_obs).reshape(-1, 1)

    return y


def get_eps():
    global total_steps
    # compute decaying eps_threshold to encourage exploration at the beginning
    return eps_end + (eps_start - eps_end) * math.exp(-1. * total_steps / eps_decay)


def choose_action(Q, observation, discrete_actions):
    eps_threshold = get_eps()
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

    obs = minibatch[:, :dim_obs]
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


def evaluate_policy_Q(Q, episodes=100, max_steps=10000, render=0):
    global env
    global discrete_actions

    print("Evaluating the policy")
    Q.eval()

    rewards = np.empty(episodes)

    for episode in range(0, episodes):
        print("\rEpisode {:4d}/{:4d}".format(episode, episodes), end="")
        total_reward = 0
        obs = env.reset()
        steps = 0
        done = False
        while steps <= max_steps and not done:
            action_idx = choose_action(Q, obs, discrete_actions)
            action = discrete_actions[action_idx]

            obs, reward, done, _ = env.step(action)

            total_reward += reward
            steps += 1

            if episode < render:
                env.render()
                if steps % 25 == 0:
                    print("\rEpisode {:4d}/{:4d} > {:4d} steps".format(episode, episodes, steps), end="")

        rewards[episode] = total_reward
    print("\rDone.")

    print("Stats: Episodes {}, avg: {}, std: {}".format(episodes, rewards.mean(), rewards.std()))


memory = ReplayMemory(8000, transition_size)
# The amount of random samples gathered before the learning starts (should be <= capacity of replay memory)
create_initial_samples(env, memory, 2000, discrete_actions)

# the size of the sampled minibatch used for learning in each step
minibatch_size = 128

# init NNs for Q function approximation
Q = DQNModel(n_inputs=dim_obs, n_outputs=len(discrete_actions), optimizer="rmsprop", lr=1e-2)
target_Q = DQNModel(n_inputs=dim_obs, n_outputs=len(discrete_actions), optimizer=None)
target_Q.load_state_dict(Q.state_dict())

scheduler = StepLR(Q.optimizer, 200, 0.99)

# the amount of steps after which the target model is updated
target_model_update_steps = 1000

# how many training episodes to do
max_episodes = 5000

# the maximum number of steps per episode
max_steps_per_episode = 15000
# stop the current episode after this many steps..
soft_max_episode_steps = 2500
# ..when the current episode performed worse than this avg reward threshold
soft_avg_reward_threshold = 0.8

episodes = 0
total_steps = 0

episode_steps = 0
episode_reward = 0
episode_loss = 0

# don't render all episodes
render_episodes_mod = 10

obs = env.reset()
# try-catch block to allow stopping the training process on KeyboardInterrupt
try:
    while episodes < max_episodes:
        scheduler.step()
        Q.eval()

        action_idx = choose_action(Q, obs, discrete_actions)
        action = discrete_actions[action_idx]

        next_obs, reward, done, _ = env.step(action)
        # reward clipping
        # reward = min(max(-1., reward), 1.)

        episode_reward += reward

        # save the observed step in our replay memory
        memory.push((*obs, *action_idx, reward, *next_obs, done))

        # remember last observation
        obs = next_obs

        if episodes % render_episodes_mod == 0:
            env.render()

        loss = optimize(memory, Q, target_Q, criterion=nn.SmoothL1Loss())
        episode_loss += loss

        writer.add_scalar("loss", loss, total_steps)

        total_steps += 1
        episode_steps += 1

        # copy the current model each target_model_update_steps steps
        if total_steps % target_model_update_steps == 0:
            target_Q.load_state_dict(Q.state_dict())

        avg_reward = episode_reward / episode_steps

        if episode_steps % 100 == 0:
            print("\rEpisode {:5d} -- total steps: {:8d} > steps: {:4d} (Running)"
                  .format(episodes + 1, total_steps, episode_steps), end="")

        if done or (episode_steps >= soft_max_episode_steps and avg_reward < soft_avg_reward_threshold) \
                or episode_steps >= max_steps_per_episode:
            obs = env.reset()
            episodes += 1

            # save model after each epoch
            # save_checkpoint({
            #     'epoch': total_steps,
            #     'episodes': episodes,
            #     'Q': Q.state_dict(),
            #     'target_Q': target_Q.state_dict(),
            #     'reward': episode_reward,
            #     'optimizer': Q.optimizer.state_dict(),
            # }, filename="./checkpoints/Q_model_epoch-{}_reward-{}.pth.tar".format(total_steps, episode_reward))

            print("\rEpisode {:5d} -- total steps: {:8d} > avg reward: {:.10f} -- steps: {:4d} -- reward: {:5.5f} "
                  "-- training loss: {:10.5f} -- lr: {:0.8f} -- eps: {:0.8f}"
                  .format(episodes, total_steps, avg_reward, episode_steps, episode_reward, episode_loss,
                          get_current_lr(Q.optimizer), get_eps()))

            writer.add_scalar("avg_reward", avg_reward, episodes)
            writer.add_scalar("total_reward", episode_reward, episodes)
            Q.eval()
            writer.add_scalar("first_best_value", get_best_values(Q, np.atleast_2d(obs))[0], episodes)

            episode_steps = 0
            episode_loss = 0
            episode_reward = 0

except(KeyboardInterrupt):
    print()
    evaluate_policy_Q(target_Q, episodes=200, render=5)
except:
    raise

env.close()
