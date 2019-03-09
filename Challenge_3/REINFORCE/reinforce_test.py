import torch

import gym
import quanser_robots
import numpy as np

from Challenge_3.Policy.ContinuousPolicy import ContinuousPolicy
from Challenge_3.REINFORCE.reinforce import REINFORCE
from Challenge_3.Policy.DiscretePolicy import DiscretePolicy

seed = 42

# env = gym.make("Pendulum-v0")
from Challenge_3.Util import get_samples

env = gym.make("Levitation-v1")
# env = gym.make("CartpoleStabShort-v0")

torch.manual_seed(seed)
np.random.seed(seed)
env.seed(seed)

# print_random_policy_reward(env, episodes=1)
discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 2)
reinforce_model = DiscretePolicy(env, discrete_actions, n_hidden_units=8)
#reinforce_model = ContinuousPolicy(env, n_hidden_units=6, state_dependent_sigma=False)

low = env.observation_space.low
low[1] = 0

reinforce = REINFORCE(env, reinforce_model, gamma=1, lr=0.1, normalize_observations=False, low=low)
reinforce.train(min_steps=100, save_best=True, render_episodes_mod=100, max_episodes=10)

# load the best model again
reinforce_model.load_state_dict(torch.load("../checkpoints/reinforce_best_weights.pth"))

sample_episodes, sample_steps, memory, episode_rewards, episode_steps = \
                    get_samples(env, reinforce_model, min_steps=50000, normalize_observations=False, low=None, high=None)

print("Eval ({} episodes): {:.4f} +/- {:.4f} ({:.4f} +/- {:.4f} steps)".
      format(sample_episodes, episode_rewards.mean(), episode_rewards.std(), episode_steps.mean(), episode_steps.std()))