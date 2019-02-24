import gym
import quanser_robots

from Challenge_3.Policy.ContinuousPolicy import ContinuousPolicy
from Challenge_3.REINFORCE.reinforce import REINFORCE
from Challenge_3.Util import make_env_step_silent

env = gym.make("BallBalancerSim-v0")

make_env_step_silent(env)

# print_random_policy_reward(env, episodes=30)

policy = ContinuousPolicy(env, n_hidden_units=12)
reinforce = REINFORCE(env, policy, 0.99, 1e-4, normalize_observations=True)
reinforce.train(save_best=False, render_episodes_mod=300, max_episodes=10000)

env.close()