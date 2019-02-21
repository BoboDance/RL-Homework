import gym
import quanser_robots
import numpy as np

from Challenge_3.REINFORCE.reinforce import REINFORCE
from Challenge_3.REINFORCE.reinforce_model import REINFORCEModel
from Challenge_3.Util import print_random_policy_reward

# env = gym.make("Pendulum-v0")
env = gym.make("Levitation-v0")

# print_random_policy_reward(env, episodes=2)

discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 4)
reinforce_model = REINFORCEModel(env, discrete_actions.size)
reinforce = REINFORCE(env, reinforce_model, discrete_actions, 0.99, 1e-3)
reinforce.train(save_best=False, render_episodes_mod=100, max_episodes=1000)