import copy
import os
import sys
from collections import deque

import numpy as np
import torch
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


def print_random_policy_reward(env, episodes=10):
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
    """
    Gather the returns from the back to the front using the given (subsequent) rewards.

    :param rewards: The sampled rewards
    :param gamma: The discount factor
    :param dones: Whether the episode was done after this step
    :param normalize: Whether the returns should be normalized in the end
    :return: the returns in a torch tensor
    """
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
    """
    Gather samples from the environment.

    :param env: The environment used for sampling
    :param policy: The policy which defines which actions will be used
    :param min_steps: The minimum amount of steps which will be sampled (can be multiple episodes)
    :param max_episode_steps: The maximum amount of steps for a single sample episode
    :param normalize_observations: Whether to normalize observations
    :param low: The lower observation bound for normalization (use None to use the env.observation_space default)
    :param high: The higher observation bound for normalization (use None to use the env.observation_space default)
    :return: Sampling data and statistics about the individual episodes
    """
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

            # Hotfix because Levitation-v1 returns numpy array instead of single value
            if type(reward) is np.ndarray:
                reward = reward[0]

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


def eval_policy_fun(env, policy_fun, episodes, max_episode_steps=1000000, normalize_observations=False, low=None, high=None):
    """
    Gather samples from the environment.

    :param env: The environment used for sampling
    :param policy_fun: The policy which defines which actions will be used (only observation -> action)
    :param episodes: Number of episodes for evaluation
    :param max_episode_steps: The maximum amount of steps for a single sample episode
    :param normalize_observations: Whether to normalize observations
    :param low: The lower observation bound for normalization (use None to use the env.observation_space default)
    :param high: The higher observation bound for normalization (use None to use the env.observation_space default)
    :return: Statistics about the individual episodes
    """

    episode_rewards = deque()
    total_episodes = 0
    total_steps = 0
    episode_steps = deque()
    for episode in range(0, episodes):
        episode_reward = 0
        episode_step = 0

        state = env.reset()
        if normalize_observations:
            state = normalize_state(env, state, low=low, high=high)

        # Sample one episode
        for episode_step in range(1, max_episode_steps + 1):
            next_state, reward, done, _ = env.step(policy_fun(state))
            if normalize_observations:
                next_state = normalize_state(env, next_state, low=low, high=high)

            # Hotfix because Levitation-v1 returns numpy array instead of single value
            if type(reward) is np.ndarray:
                reward = reward[0]

            episode_reward += reward
            state = next_state

            if done:
                break

        total_episodes += 1
        total_steps += episode_step
        episode_rewards.append(episode_reward)
        episode_steps.append(episode_step)

    return total_episodes, total_steps, np.array(episode_rewards), np.array(episode_steps)


def get_reward(weights, model, env, render=False, return_steps=False):
    cloned_model = nes_load_model_weights(weights, model)

    state = env.reset()
    done = False
    total_reward = 0
    t = 0
    while not done:
        if render:
            env.render()

        with torch.no_grad():
            action = cloned_model(torch.Tensor(state)).numpy()

        state, reward, done, _ = env.step(action)
        total_reward += reward
        t += 1

    if return_steps:
        return total_reward, t

    return total_reward


def nes_load_model_weights(weights, model):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)
    return cloned_model