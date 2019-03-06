import gym
import quanser_robots
import numpy as np

from Challenge_3.Policy.ContinuousPolicy import ContinuousPolicy
from Challenge_3.REINFORCE.reinforce import REINFORCE
from Challenge_3.Policy.DiscretePolicy import DiscretePolicy

# env = gym.make("Pendulum-v0")
env = gym.make("Levitation-v1")
# env = gym.make("CartpoleStabShort-v0")

# print_random_policy_reward(env, episodes=1)
# discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 5)
# reinforce_model = DiscretePolicy(env, discrete_actions, n_hidden_units=16)
reinforce_model = ContinuousPolicy(env, n_hidden_units=10, state_dependent_sigma=False)

low = env.observation_space.low
low[1] = 0

reinforce = REINFORCE(env, reinforce_model, 0.90, 1e-3, normalize_observations=False, low=low)
reinforce.train(save_best=False, render_episodes_mod=100, max_episodes=100000)
