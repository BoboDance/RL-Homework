import logging
import pickle
import time
from functools import partial

import gym
import quanser_robots
from gym import logger as gym_logger

from Challenge_3.NES.nes import NES
from Challenge_3.Policy.NESPolicy import NESPolicy
from Challenge_3.Util import make_env_step_silent, get_reward

quanser_robots
gym_logger.setLevel(logging.CRITICAL)

env_name = "BallBalancerSim-v0"

env = gym.make(env_name)
# Disable cancerous outputs in order to see interesting stuff
make_env_step_silent(env)

print(f"Starting run for NES on {env_name}.")

# important parameters
population_size = 17
sigma = 1
lr = 5e-2

# early stopping if goal is reached for n steps
reward_goal = 700
consecutive_goal_stopping = 20

thread_count = 2
render = False
iterations = 10000
normalize_rewards = True

# decay for lr and exploration
decay = 1
sigma_decay = 1

model = NESPolicy(env, n_hidden_units=10)

partial_func = partial(get_reward, model=model, env=env)
global_parameters = list(model.parameters())

nes = NES(global_parameters, partial_func, population_size=population_size, sigma=sigma, learning_rate=lr,
          reward_goal=reward_goal, consecutive_goal_stopping=consecutive_goal_stopping, threadcount=thread_count,
          render_test=render, decay=decay, sigma_decay=sigma_decay, normalize_reward=normalize_rewards)

start = time.time()
final_weights = nes.run(iterations=iterations, print_mod=1)
end = time.time() - start

reward = partial_func(final_weights, render=render)
pickle.dump(final_weights, open(f"../checkpoints/reward-{reward}.pkl", 'wb'))

print(f"Reward from final weights: {reward}")
print(f"Time to completion: {end}")


# Best setup so far.
# NN architecture:
# Sequential(
#   (0): Linear(in_features=8, out_features=10, bias=True)
#   (1): ReLU()
#   (2): Linear(in_features=10, out_features=10, bias=True)
#   (3): ReLU()
#   (4): Linear(in_features=10, out_features=2, bias=True)
# )
# Parameter Settings:
# {   'consecutive_goal_count': 0,
#     'consecutive_goal_stopping': 20,
#     'decay': 0.99,
#     'lr': 0.05,
#     'normalize_reward': True,
#     'pop_size': 20,
#     'render_test': False,
#     'reward_goal': 700,
#     'sigma': 1,
#     'sigma_decay': 0.99}