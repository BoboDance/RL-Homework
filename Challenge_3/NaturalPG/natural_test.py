import torch

import gym
import numpy as np
import quanser_robots

from Challenge_3.NaturalPG.NaturalPG import NaturalPG
from Challenge_3.NaturalPG.ValueModel import ValueModel
from Challenge_3.Policy.ContinuousPolicy import ContinuousPolicy
from Challenge_3.Util import make_env_step_silent

env = gym.make("BallBalancerSim-v0")
# env = gym.make("Pendulum-v0")
# make_env_step_silent(env)
seed = 1
if seed is not None:
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# print_random_policy_reward(env, episodes=30)

actor = ContinuousPolicy(env, n_hidden_units=16)
# critic = ValueModel(env, n_hidden_units=64)
critic = None

naturalPG = NaturalPG(env, actor, gamma=0.99, critic=critic, use_tensorboard=False)
naturalPG.train(max_episodes=2000)

env.close()