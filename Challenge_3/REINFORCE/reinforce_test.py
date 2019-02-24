import gym
import quanser_robots
import numpy as np

from Challenge_3.REINFORCE.reinforce import REINFORCE
from Challenge_3.REINFORCE.reinforce_model import REINFORCEModel
from Challenge_3.Util import print_random_policy_reward

# env = gym.make("Pendulum-v0")
env = gym.make("Levitation-v1")

# print_random_policy_reward(env, episodes=1)

discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 5)
reinforce_model = REINFORCEModel(env, discrete_actions.size, n_hidden_units=16)
low = env.observation_space.low
low[1] = 0
reinforce = REINFORCE(env, reinforce_model, discrete_actions, 0.93, 1e-4, normalize_observations=True, low = low)
reinforce.train(save_best=False, render_episodes_mod=100, max_episodes=10000)