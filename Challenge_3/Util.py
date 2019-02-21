import gym
import numpy as np


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