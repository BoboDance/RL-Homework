import os
import sys
from collections import deque

import torch

import gym
import numpy as np
from torch import nn


def normalize_state(env, observation, low=None, high=None):
    """
    Normalize a given observation using the observation space boundaries from the environment or manual limits.

    :param env: gym env to work with
    :param observation: state observation
    :param low: manual value for the lower observation space limit
    :param high: manual value for the higher observation space limit
    :return: the normalized observation value
    """

    if low is None:
        low = env.observation_space.low

    if high is None:
        high = env.observation_space.high

    return (observation - low) / (high - low)


def print_random_policy_reward(env, episodes = 10):
    rewards = np.zeros(episodes)

    print("Calculating random policy rewards.. ", end='')

    for episode in range(0, episodes):
        env.reset()
        done = False
        episode_reward = 0
        step = 0
        while not done:
            obs, reward, done, info = env.step(env.action_space.sample())
            episode_reward += reward
            step += 1

            if step % 100 == 0:
                print("\rCalculating random policy rewards.. Episode {}, Step {}".format(episode, step), end='')

        rewards[episode] = episode_reward

    print("\rRandom policy: {:.4f} +/- {:.4f}".format(rewards.mean(), rewards.std()))


def make_env_step_silent(env):
    """
    Makes the step function of the given environment silent.

    :param env: The environment which is too verbose
    :return: The old step function (can be used to make it verbose again)
    """
    original_step = env.step
    original_stdout = sys.stdout
    null_stdout = open(os.devnull, 'w')

    def silent_step(action):
        sys.stdout = null_stdout
        values = original_step(action)
        sys.stdout = original_stdout
        return values

    env.step = silent_step

    # close access to devnull in the end
    original_close = env.close
    env.close = lambda: {original_close, null_stdout.close()}

    return original_step


def init_weights(m):
    """
    Initializes the weights of the given model by using kaiming normal initialization.
    :param m: Handle for the model
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


def get_returns_torch(rewards, gamma, dones, normalize=True):
    returns = torch.zeros(rewards.shape)
    dones = (1 - dones)
    accumulated_return = 0

    i = len(rewards) - 1
    for reward in rewards[::-1]:
        accumulated_return = reward + gamma * accumulated_return * dones[i]
        returns[i] = accumulated_return
        i -= 1

    if normalize:
        returns = (returns - returns.mean()) / returns.std()

    return returns


def get_samples(env, policy, min_steps, max_episode_steps=1000000, normalize_observations=False, low=None, high=None):
    memory = deque()

    episode_rewards = deque()
    total_episodes = 0
    total_steps = 0
    episode_steps = deque()
    while True:
        episode_reward = 0
        episode_step = 0

        state = env.reset()
        if normalize_observations:
            state = normalize_state(env, state, low=low, high=high)

        # Sample one episode
        for episode_step in range(1, max_episode_steps + 1):
            action, log_confidence = policy.choose_action_by_sampling(state)

            next_state, reward, done, _ = env.step(action)
            if normalize_observations:
                next_state = normalize_state(env, next_state, low=low, high=high)

            memory.append([state, action, reward, done, log_confidence])

            episode_reward += reward
            state = next_state

            if done:
                break

        total_episodes += 1
        total_steps += episode_step
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

        # Check whether we need more episodes to get above min_steps
        if total_steps >= min_steps:
            break

    return total_episodes, total_steps, np.array(memory), np.array(episode_rewards), np.array(episode_steps)
