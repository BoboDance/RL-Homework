import gym
import quanser_robots
import numpy as np

from Challenge_3.Util import print_random_policy_reward, make_env_step_silent

env = gym.make("BallBalancerSim-v0")

make_env_step_silent(env)

print_random_policy_reward(env, episodes=30)

env.close()