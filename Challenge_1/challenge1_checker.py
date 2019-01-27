"""
Use this file to check that your implementation complies with our evaluation
interface.
"""

import gym
from gym.wrappers.monitor import Monitor
from Challenge_1.challenge1 import get_model, get_policy
import quanser_robots
import numpy as np
import logging

# avoid auto removal of import with pycharm
quanser_robots

logging.info('script start')

# 1. Learn the model f: s, a -> s', r
env_name = 'Pendulum-v0'
#env_name = 'Qube-v0'
env = Monitor(gym.make(env_name), 'training', video_callable=False, force=True)
env.seed(98251624)

max_num_samples = 10000
model = get_model(env, max_num_samples)
env.close()

# Your model will be tested on the quality of prediction
obs = env.reset()
act = env.action_space.sample()
nobs, rwd, _, _ = env.step(act)
nobs_pred, rwd_pred = model(obs, act)
print(f'truth = {nobs, rwd}\nmodel = {nobs_pred, rwd_pred}')

# 2. Perform dynamic programming using the learned model
env = Monitor(gym.make(env_name), 'evaluation', force=True)
#env = gym.make(env_name)
env.seed(31186490)
policy = get_policy(model, env.observation_space, env.action_space)

# Your policy will be judged based on the average episode return
n_eval_episodes = 100
lengths = np.zeros(n_eval_episodes)
rewards = np.zeros(n_eval_episodes)
for i in range(n_eval_episodes):
    done = False
    obs = env.reset()
    while not done:
        act = policy(obs)
        obs, reward, done, _ = env.step(act)
        lengths[i] += 1
        rewards[i] += reward
        if i <= 2:
            env.render()

av_ep_ret = sum(env.get_episode_rewards()) / len(env.get_episode_rewards())
print('average reward over %d episodes: %.3f +- %.3f min: %.3f max: %.3f'
      % (n_eval_episodes, rewards.mean(), rewards.std(), rewards.min(), rewards.max()))
print(f'average return per episode: {av_ep_ret}')
print(f'average length per episode: {lengths.mean()}')

logging.info('script end')
env.close()
