import math
import gym
import numpy as np
import scipy.linalg
import sys
import quanser_robots
import pickle

from Challenge_2.LSPI.BasisFunctions.FourierBasis import FourierBasis
from Challenge_2.LSPI.LSPI import LSPI
from Challenge_2.LSPI.Policy import Policy
from Challenge_2.LSPI.BasisFunctions.RadialBasisFunction import RadialBasisFunction
from Challenge_2.Common.ReplayMemory import ReplayMemory
from Challenge_2.Common.Util import create_initial_samples, normalize_state, evaluate
from Challenge_2.LSPI.Util import get_policy_fun

seed = 2
np.random.seed(seed)
env = gym.make("CartpoleStabShort-v0")
env.seed(seed)
dim_obs = env.observation_space.shape[0]

discrete_actions = np.linspace(-5, 5, 3)

# Basis function and policy

n_features = 100
# TODO: find better init
# RBFS base function
# means = np.random.multivariate_normal((low + high) / 2, np.diag(high / 3), size=(n_features,))
# means = np.random.uniform(low, high, size=(n_features, dim_obs))
# means = np.array([np.linspace(low[i], high[i], n_features) for i in range(dim_obs)]).T
# means = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
# means = np.array(np.meshgrid(*tuple([np.linspace(low[i], high[i], 2) for i in range(dim_obs)]))).T.reshape(-1, dim_obs)
# beta = .8  # parameter for width of gaussians
# basis_function = RadialBasisFunction(input_dim=dim_obs, means=means, n_actions=len(discrete_actions), beta=beta)

# Fourier base function
basis_function = FourierBasis(input_dim=dim_obs, n_features=n_features, n_actions=len(discrete_actions))
low = np.array(list(env.observation_space.low[:3]) + [-2.5, -30])
high = np.array(list(env.observation_space.high[:3]) + [2.5, 30])

policy = Policy(basis_function=basis_function, n_actions=len(discrete_actions), eps=0)

normalize = True

lspi = LSPI(env=env, policy=policy, discrete_actions=discrete_actions, normalize=normalize, low=low, high=high,
            gamma=0.99, theta=1e-5, samples_count=25000)
lspi.train(policy_step_episodes=1, do_render=False)
# policy = pickle.load(open("policy.pkl", "rb"))

mean_reward = evaluate(env, get_policy_fun(env, policy, normalize, discrete_actions), episodes=10)
pickle.dump(policy, open("policy_{:.4f}.pkl".format(mean_reward), "wb"))
