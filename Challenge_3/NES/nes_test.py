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

# general parameters
population_size = 20
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
decay = .99
sigma_decay = .99

model = NESPolicy(env, n_hidden_units=25)

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
