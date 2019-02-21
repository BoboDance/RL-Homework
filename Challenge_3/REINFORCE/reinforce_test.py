import gym
import quanser_robots
import numpy as np

from Challenge_3.REINFORCE.reinforce import REINFORCE
from Challenge_3.REINFORCE.reinforce_model import REINFORCEModel
from Challenge_3.Util import print_random_policy_reward

env = gym.make("Levitation-v0")

# print_random_policy_reward(env, episodes=2)

reinforce_model = REINFORCEModel()
reinforce = REINFORCE(env, reinforce_model, np.linspace(env.action_space.low, env.action_space.high, 4), 0.99, 1e-4, 100)
reinforce.train(save_best=False)