# -*- coding: utf-8 -*-
"""Contains main interface to LSPI algorithm."""
import math
import quanser_robots
import gym
import numpy as np
import scipy

from Challenge_2.LSPI.Policy import Policy
from Challenge_2.LSPI.RBF import RBF
from Challenge_2.LSPI.ReplayMemory import ReplayMemory
from Challenge_2.LSPI.Util import create_initial_samples

seed = 1
np.random.seed(seed)

env = gym.make("CartpoleStabShort-v0")
env.seed(seed)

dim_obs = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]

discrete_actions = np.linspace(-2.5, 2.5, 11)
# discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 20)
print("Used discrete actions: ", discrete_actions)

# transition: observation, action, reward, next observation, done
transition_size = dim_obs + dim_action + 1 + dim_obs + 1

# Helps to ensure few samples give a matrix of full rank, choose 0 if not desired

ACTION_IDX = dim_obs
REWARD_IDX = dim_obs + dim_action
NEXT_OBS_IDX = dim_obs + dim_action + 1
DONE_IDX = -1

importance_weights = False

# the probability to choose an random action decaying over time
eps_start = .25
eps_end = 0.05
eps_decay = 100

gamma = 0.99  # discount
theta = 1e-5  # convergence criterion

max_episodes = 10000
minibatch_size = 256

n_features = 100
beta = 4  # parameter for width of gaussians


def LSTDQ_iteration(samples, policy, precondition_value=.1):
    """
    Compute Q value function of current policy via LSTDQ iteration.
    If a matrix has full rank: scipy.linalg solver
    else: Least squares solver
    :param samples: data samples
    :param policy: policy to work with
    :param precondition_value: Helps to ensure few samples give a matrix of full rank, choose 0 if not desired
    :return:
    """
    global ACTION_IDX
    global REWARD_IDX
    global NEXT_OBS_IDX
    global DONE_IDX

    k = policy.basis_function.size()

    obs = samples[:, 0: ACTION_IDX]
    action_idx = samples[:, ACTION_IDX]
    reward = samples[:, REWARD_IDX: NEXT_OBS_IDX]
    next_obs = samples[:, NEXT_OBS_IDX: DONE_IDX]
    done = samples[:, DONE_IDX].astype(np.bool)

    phi = (policy.basis_function(obs, action_idx).reshape((-1, len(samples))))
    phi_next = np.zeros((k, len(samples)))

    if np.any(~done):
        sel_next_obs = next_obs[~done]
        best_action = policy.get_best_action(sel_next_obs)
        phi_next[:, ~done] = (policy.basis_function(sel_next_obs, best_action).reshape((-1, len(sel_next_obs))))

    A = phi.dot((phi - policy.gamma * phi_next).T) + np.identity(k) * precondition_value
    b = phi @ reward

    rank_A = np.linalg.matrix_rank(A)

    if rank_A == k:
        w = scipy.linalg.solve(A, b)
    else:
        print(f'A matrix does not have full rank {k} > {rank_A}. Using least squares solver.')
        w = scipy.linalg.lstsq(A, b)[0]

    phi.dot((phi - policy.gamma * phi_next).T)
    return w.reshape((-1,))


memory = ReplayMemory(5000, transition_size)
# The amount of random samples gathered before the learning starts (should be <= capacity of replay memory)
create_initial_samples(env, memory, 500, discrete_actions)

low = np.array(list(env.observation_space.low[:3]) + [-5, -5])
high = np.array(list(env.observation_space.high[:3]) + [5, 5])
means = np.random.multivariate_normal((low - high) / 2, np.diag(high / 3), size=(n_features,))
# means = np.array([np.linspace(low[i], high[i], n_features) for i in range(dim_obs)]).T

basis_function = RBF(input_dim=dim_obs, means=means, n_actions=len(discrete_actions), beta=beta)
policy = Policy(basis_function=basis_function, n_actions=len(discrete_actions), gamma=gamma, eps=eps_start)

delta = np.inf
episodes = 0
total_steps = 0

episode_reward = 0
episode_steps = 0

obs = env.reset()

while delta >= theta and episodes <= max_episodes:

    total_steps += 1
    episode_steps += 1

    action_idx = policy.choose_action(obs)
    action = discrete_actions[action_idx]

    next_obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()
        episodes += 1
        print(
            "Episode {:5d} -- total steps: {:8d} > avg reward: {:.10f} -- episode steps: {:4d} "
            "-- episode reward: {:5.5f} -- delta: {} -- epsilon {}".format(
                episodes, total_steps, reward / episode_steps, episode_steps, episode_reward, delta, policy.eps))
        episode_steps = 0
        episode_reward = 0

        # reduce random action prob each episode
        policy.eps = eps_end + (eps_start - eps_end) * math.exp(-1. * episodes / eps_decay)

    reward = min(max(-1., reward), 1.)
    episode_reward += reward

    memory.push((*obs, *action_idx, reward, *next_obs, done))

    obs = next_obs
    env.render()

    if total_steps % 50 == 0:
        if importance_weights:
            raise NotImplementedError()
        else:
            samples = memory.sample(minibatch_size)
            new_weights = LSTDQ_iteration(samples, policy)

        delta = np.linalg.norm(new_weights - policy.w)
        policy.w = new_weights
