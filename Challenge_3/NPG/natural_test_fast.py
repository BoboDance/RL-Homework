import torch

import gym
import numpy as np
import quanser_robots

from Challenge_3.NPG.NaturalPG import NaturalPG
from Challenge_3.NPG.ValueModel import ValueModel
from Challenge_3.Policy.ContinuousPolicy import ContinuousPolicy
from Challenge_3.Util import make_env_step_silent, get_samples, set_seed, eval_policy_fun

env = gym.make("BallBalancerSim-v0")
make_env_step_silent(env)

seed = 7
set_seed(env, seed)

actor = ContinuousPolicy(env, n_hidden_units=32, state_dependent_sigma=False)
# critic = ValueModel(env, n_hidden_units=64)
critic = None

naturalPG = NaturalPG(env, actor, gamma=0.99, critic=critic, use_tensorboard=True)
naturalPG.train(min_steps=1500, max_episodes=100)

# load the best model again
actor.load_state_dict(torch.load("../checkpoints/npg_actor_best_weights.pth"))


def policy_fun(obs):
    action, _ = actor.choose_action_by_sampling(obs)
    return action


sample_episodes, sample_steps, episode_rewards, episode_steps = \
                    eval_policy_fun(env, policy_fun, 25, normalize_observations=False, low=None, high=None)

print("Eval ({} episodes): {:.4f} +/- {:.4f} ({:.4f} +/- {:.4f} steps)".
      format(sample_episodes, episode_rewards.mean(), episode_rewards.std(), episode_steps.mean(), episode_steps.std()))

env.close()