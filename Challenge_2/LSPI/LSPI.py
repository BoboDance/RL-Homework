# -*- coding: utf-8 -*-
"""Contains main interface to LSPI algorithm."""
import math
import gym
import numpy as np
import scipy.linalg
import quanser_robots

from Challenge_2.LSPI.BasisFunctions.FourierBasis import FourierBasis
from Challenge_2.LSPI.Policy import Policy
from Challenge_2.LSPI.BasisFunctions.RadialBasisFunction import RadialBasisFunction
from Challenge_2.Common.ReplayMemory import ReplayMemory
from Challenge_2.Common.Util import create_initial_samples, normalize_state

seed = 2
np.random.seed(seed)

# env = gym.make("Pendulum-v0")
env = gym.make("CartpoleStabShort-v0")
# env = gym.make("CartPole-v0")
env.seed(seed)

dim_obs = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]
# dim_action = 1


# discrete_actions = np.arange(env.action_space.n)
discrete_actions = np.linspace(-5, 5, 3)
# discrete_actions = np.linspace(env.action_space.low, env.action_space.high, 5)
print("Used discrete actions: ", discrete_actions)

# transition: observation, action, reward, next observation, done
transition_size = dim_obs + dim_action + 1 + dim_obs + 1

ACTION_IDX = dim_obs
REWARD_IDX = dim_obs + dim_action
NEXT_OBS_IDX = dim_obs + dim_action + 1
DONE_IDX = -1

importance_weights = False

# the probability to choose an random action decaying over time
eps_start = 0  # .25
eps_end = 0  # .05
eps_decay = 100

gamma = 0.99  # discount
theta = 1e-5  # convergence criterion

# sample hyperparameter
max_episodes = 20000
replay_memory_size = 10000
initial_samples = 10000
minibatch_size = 10000
optimize_after_steps = 1

n_features = 100  # current best: RBF with 20 features and beta .8

# RBFS base function
beta = .8  # parameter for width of gaussians

# Fourier base function
width = 2.5  # width of fourier features

do_render = True
normalize = True


def LSTDQ_iteration(samples, policy, precondition_value=.01, use_optimized=False):
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

    phi = policy.basis_function(obs, action_idx)
    phi_next = np.zeros_like(phi)

    if np.any(~done):
        sel_next_obs = next_obs[~done]
        best_action = policy.get_best_action(sel_next_obs)
        phi_next[~done, :] = policy.basis_function(sel_next_obs, best_action)

    if not use_optimized:
        A = (phi.T @ (phi - gamma * phi_next) + np.identity(k) * precondition_value) / minibatch_size
        b = (phi.T @ reward) / minibatch_size

        # a_mat, b_vec, phi_sa, phi_sprime = loop_it(samples, precondition_value)
        rank_A = np.linalg.matrix_rank(A)

        if rank_A == k:
            w = scipy.linalg.solve(A, b)
        else:
            # print(f'A matrix does not have full rank {k} > {rank_A}. Using least squares solver.')
            w = scipy.linalg.lstsq(A, b)[0]

    else:
        B = (1 / precondition_value) * np.identity(k)
        b = 0
        for i in range(len(phi)):
            p = phi[i].reshape(-1, 1)
            pn = phi_next[i].reshape(-1, 1)

            top = B @ (p @ (p - gamma * pn).T) @ B.T
            bottom = 1 + (p - gamma * pn).T @ B @ p
            B -= top / bottom
            b += p @ reward[i]

        w = B @ b

    return w.reshape((-1,))


def loop_it(samples, precondition_value):
    k = policy.basis_function.size()

    a_mat = np.zeros((k, k))
    np.fill_diagonal(a_mat, precondition_value)

    b_vec = np.zeros((k, 1))

    phi = []
    phi_next = []

    for sample in samples:

        obs = sample[0: ACTION_IDX]
        action_idx = sample[ACTION_IDX]
        reward = sample[REWARD_IDX: NEXT_OBS_IDX]
        next_obs = sample[NEXT_OBS_IDX: DONE_IDX]
        done = sample[DONE_IDX].astype(np.bool)

        phi_sa = (policy.basis_function.evaluate(obs, action_idx).reshape((-1, 1)))

        if not done:
            best_action = policy.best_action(next_obs)
            phi_sprime = (policy.basis_function.evaluate(next_obs, best_action).reshape((-1, 1)))
        else:
            phi_sprime = np.zeros((k, 1))

        phi.append(phi_sa)
        phi_next.append(phi_sprime)

        a_mat += phi_sa.dot((phi_sa - gamma * phi_sprime).T)
        b_vec += phi_sa * reward

    return a_mat, b_vec, np.array(phi), np.array(phi_next)


# max_episodes = 50

# for i in range(30):

# print("-" * 100)
# print(f"Test run {i} started.")
# Fourier base function
# width = np.random.uniform(.01, 10)  # width of fourier features
# n_features = np.random.randint(10, 50)
# print(f"Fourier width: {width}, n_features {n_features}")

memory = ReplayMemory(replay_memory_size, transition_size)
# The amount of random samples gathered before the learning starts (should be <= capacity of replay memory)
create_initial_samples(env, memory, initial_samples, discrete_actions, normalize=normalize)

low = np.array(list(env.observation_space.low[:3]) + [-2.5, -30])
high = np.array(list(env.observation_space.high[:3]) + [2.5, 30])

# low = env.observation_space.low
# high = env.observation_space.high

# TODO: find better init
# means = np.random.multivariate_normal((low + high) / 2, np.diag(high / 3), size=(n_features,))
# means = np.random.uniform(low, high, size=(n_features, dim_obs))
# means = np.array([np.linspace(low[i], high[i], n_features) for i in range(dim_obs)]).T
# means = np.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
# means = np.array(np.meshgrid(*tuple([np.linspace(low[i], high[i], 2) for i in range(dim_obs)]))).T.reshape(-1, dim_obs)
# basis_function = RadialBasisFunction(input_dim=dim_obs, means=means, n_actions=len(discrete_actions), beta=beta)

basis_function = FourierBasis(input_dim=dim_obs, n_features=n_features, n_actions=len(discrete_actions))
policy = Policy(basis_function=basis_function, n_actions=len(discrete_actions), eps=eps_start)

delta = np.inf
episodes = 0
total_steps = 0

episode_reward = 0
episode_steps = 0

obs = env.reset()

while delta >= theta and episodes <= max_episodes:

    total_steps += 1
    episode_steps += 1

    if normalize:
        obs = normalize_state(env, obs)

    action_idx = policy.choose_action(obs)
    action = discrete_actions[action_idx]

    next_obs, reward, done, _ = env.step(action)

    if done:
        obs = env.reset()
        episodes += 1
        print(
            "Episode {:5d} -- total steps: {:8d} > avg reward: {:.10f} -- episode steps: {:4d} "
            "-- episode reward: {:5.5f} -- delta: {:6.10f} -- epsilon {:.10f}".format(
                episodes, total_steps, reward / episode_steps, episode_steps, episode_reward, delta, policy.eps))
        episode_steps = 0
        episode_reward = 0

        # reduce random action prob each episode
        policy.eps = eps_end + (eps_start - eps_end) * math.exp(-1. * episodes / eps_decay)

    # reward = min(max(-1., reward), 1.)
    episode_reward += reward

    # memory.push((*obs, *action_idx, reward, *next_obs, done))

    obs = next_obs
    if do_render and episode_steps % 12 == 0:
        env.render()

    if total_steps % optimize_after_steps == 0:
        if importance_weights:
            raise NotImplementedError()
        else:
            # samples = memory.sample(minibatch_size)
            new_weights = LSTDQ_iteration(memory.memory, policy)

        delta = np.linalg.norm(new_weights - policy.w)
        print(delta)
        policy.w = new_weights

obs = env.reset()
episode_reward = 0
episode_steps = 0

done = False

while not done:
    episode_steps += 1

    action_idx = policy.choose_action(obs)
    action = discrete_actions[action_idx]

    next_obs, reward, done, _ = env.step(action)
    episode_reward += reward

    if done:
        print("Avg reward: {:.10f} -- episode steps: {:4d} -- episode reward: {:5.5f}".format(reward / episode_steps,
                                                                                              episode_steps,
                                                                                              episode_reward))
