import pickle

import numpy as np
import torch

import gym
import quanser_robots
from Challenge_3.Policy.NESPolicy import NESPolicy
from Challenge_3.Util import make_env_step_silent, nes_load_model_weights, eval_policy_fun

env = gym.make("BallBalancerSim-v0")
make_env_step_silent(env)

model = NESPolicy(env, n_hidden_units=10)
weights = pickle.load(open(f"../checkpoints/reward-715.5677933067004.pkl", 'rb'))
nes_load_model_weights(weights, model)


def policy_fun(obs):
    with torch.no_grad():
        return model(torch.Tensor(obs)).detach().numpy()


sample_episodes, sample_steps, episode_rewards, episode_steps = \
                    eval_policy_fun(env, policy_fun, 100, normalize_observations=False, low=None, high=None)

print("Eval ({} episodes): {:.4f} +/- {:.4f} ({:.4f} +/- {:.4f} steps)".
      format(sample_episodes, episode_rewards.mean(), episode_rewards.std(), episode_steps.mean(), episode_steps.std()))