import gym
import numpy as np

from Challenge_2.Common.ReplayMemory import ReplayMemory


def create_initial_samples(env: gym.Env, memory: ReplayMemory, count: int, discrete_actions: np.ndarray,
                           normalize: bool = False) -> None:
    """
    Create initial data set and push to replay memory
    :param env: gym env to work with
    :param memory: replay memory which receives data samples
    :param count: number of samples
    :param discrete_actions: valid actions to take
    :param normalize: normalize observations
    :return: None
    """
    last_observation = env.reset()
    if normalize:
        last_observation = normalize_state(env, last_observation)
    samples = 0
    while samples < count:
        # choose a random action
        action_idx = np.random.choice(range(len(discrete_actions)), 1)
        action = discrete_actions[action_idx]
        # do the step
        observation, reward, done, _ = env.step(action)
        if normalize:
            observation = normalize_state(env, observation)

        memory.push((*last_observation, *action_idx, reward, *observation, done))

        samples += 1
        last_observation = observation

        if done:
            last_observation = env.reset()


def normalize_state(env, observation):
    """
    Normlize state given max of env
    :param env: gym env to work with
    :param observation: state observation
    :return:
    """
    if env.spec.id == "CartpoleStabShort-v0":
        low = np.array(list(env.observation_space.low[:3]) + [-5, -5])
        high = np.array(list(env.observation_space.high[:3]) + [5, 5])
    else:
        low = env.observation_space.low
        high = env.observation_space.high

    return (observation - low) / (high - low)
