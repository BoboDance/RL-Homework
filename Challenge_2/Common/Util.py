import gym
import numpy as np

from Challenge_2.Common.ReplayMemory import ReplayMemory


def create_initial_samples(env: gym.Env, memory: ReplayMemory, n_samples: int, discrete_actions: np.ndarray,
                           normalize: bool = False, low = None, high = None, full_episode = False) -> None:
    """
    Create initial data set and push to replay memory
    :param env: gym env to work with
    :param memory: replay memory which receives data samples
    :param n_samples: number of samples
    :param discrete_actions: valid actions to take
    :param normalize: normalize observations
    :param low: manual low limit for observation normalization
    :param high: manual high limit for observation normalization
    :param full_episode: whether to continue the simulation until the episode has ended (does not save additional samples)
    :return: None
    """
    last_observation = env.reset()
    if normalize:
        last_observation = normalize_state(env, last_observation, low=low, high=high)

    samples_ctr = 0
    done = False
    while samples_ctr < n_samples:
        # choose a random action
        action_idx = np.random.choice(range(len(discrete_actions)), 1)
        action = discrete_actions[action_idx]
        # do the step
        observation, reward, done, _ = env.step(action)
        if normalize:
            observation = normalize_state(env, observation, low=low, high=high)

        memory.push((*last_observation, *action_idx, reward, *observation, done))

        samples_ctr += 1
        last_observation = observation

        if done:
            last_observation = env.reset()
            if normalize:
                last_observation = normalize_state(env, last_observation, low=low, high=high)

    if full_episode:
        while not done:
            # just do random actions until the episode is done
            _, _, done, _ = env.step(env.action_space.sample())


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


def evaluate(env, policy_fun, episodes=100, max_steps=10000, render=0, seed=1):
    print("Evaluating the learned policy")

    env.seed(seed)
    rewards = np.empty(episodes)
    epiode_length = np.zeros(episodes)

    for episode in range(0, episodes):
        print("\rEpisode {:4d}/{:4d}".format(episode, episodes), end="")
        total_reward = 0
        obs = env.reset()
        steps = 0
        done = False
        while steps <= max_steps and not done:
            action = policy_fun(obs)

            obs, reward, done, _ = env.step(action)

            total_reward += reward
            steps += 1

            if episode < render:
                env.render()
                if steps % 25 == 0:
                    print("\rEpisode {:4d}/{:4d} > {:4d} steps".format(episode, episodes, steps), end="")

        rewards[episode] = total_reward
        epiode_length[episode] = steps

    print("\rDone.")
    print("Statistics: {} Episodes, Total Reward: {:.4f} +/-  {:.4f}".format(episodes, rewards.mean(), rewards.std()))
    print("Episode Length: {:.4f} +/- {:.4f}".format(epiode_length.mean(), epiode_length.std()))

    return rewards.mean()