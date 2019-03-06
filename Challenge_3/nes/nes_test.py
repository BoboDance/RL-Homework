import copy
import logging
import pickle
import time
from functools import partial

import gym
import numpy as np
import quanser_robots
import torch
from gym import logger as gym_logger
from torch.autograd import Variable

from Challenge_3.Policy.ContinuousPolicy import ContinuousPolicy
from Challenge_3.Policy.NESPolicy import NESPolicy
from Challenge_3.Util import make_env_step_silent, get_reward
from Challenge_3.nes.nes import NES

quanser_robots
gym_logger.setLevel(logging.CRITICAL)

env = gym.make("BallBalancerSim-v0")
# Disable cancerous outputs in order to see interesting stuff
make_env_step_silent(env)

model = NESPolicy(env, n_hidden_units=6)

partial_func = partial(get_reward, model=model, env=env)
global_parameters = list(model.parameters())

population_size = 50
sigma = 1.
lr = 1e-3
reward_goal = 1000
consecutive_goal_stopping = 20
thread_count = 2
render = False
decay = 1.
sigma_decay = 1.
iterations = 10000

nes = NES(global_parameters, partial_func, population_size=population_size, sigma=sigma, learning_rate=lr,
          reward_goal=reward_goal, consecutive_goal_stopping=consecutive_goal_stopping, threadcount=thread_count,
          render_test=render, decay=decay, sigma_decay=sigma_decay)

start = time.time()
final_weights = nes.run(iterations=iterations, print_step=1)
end = time.time() - start

reward = partial_func(final_weights, render=True)
pickle.dump(final_weights, open(f"./checkpoints/reward-{reward}.pkl", 'wb'))

print(f"Reward from final weights: {reward}")
print(f"Time to completion: {end}")
