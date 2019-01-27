# -*- coding: utf-8 -*-
"""Contains main interface to LSPI algorithm."""
import math
import gym
import numpy as np
import scipy.linalg
import sys
import quanser_robots

from Challenge_2.LSPI.Policy import Policy
from Challenge_2.Common.ReplayMemory import ReplayMemory
from Challenge_2.Common.Util import create_initial_samples, normalize_state


class LSPI(object):

    def __init__(self, env, policy: Policy, discrete_actions, normalize, low, high, gamma, theta, samples_count, full_episode = False):
        """
        Initialize an LSPI container which can be used to find good weights for the given policy.

        :param env: the underlying environment
        :param policy: the policy which will be optimized
        :param discrete_actions: discrete actions numpy array
        :param normalize: whether to normalize the observations
        :param low: low limits for the observation space (used for normalization)
        :param high: high limits for the observation space (used for normalization)
        :param gamma: discount factor
        :param theta: convergence criterion for training
        :param samples_count: the number of samples to train on
        :param full_episode: whether the initial sampling should do a full episode (needed for monitor to work..)
        """

        self.env = env
        self.normalize = normalize
        self.gamma = gamma
        self.theta = theta

        self.dim_obs = env.observation_space.shape[0]
        self.dim_action = env.action_space.shape[0]

        self.discrete_actions = discrete_actions
        print("Used discrete actions: ", discrete_actions)

        self.low = low
        self.high = high

        # transition: observation, action, reward, next observation, done
        self.transition_size = self.dim_obs + self.dim_action + 1 + self.dim_obs + 1

        self.ACTION_IDX = self.dim_obs
        self.REWARD_IDX = self.dim_obs + self.dim_action
        self.NEXT_OBS_IDX = self.dim_obs + self.dim_action + 1
        self.DONE_IDX = -1

        # use the replay memory to store our samples
        self.memory = ReplayMemory(samples_count, self.transition_size)
        print("Creating samples...", end="")
        sys.stdout.flush()
        create_initial_samples(env, self.memory, samples_count, discrete_actions,
                               normalize=normalize, low=self.low, high=self.high, full_episode=full_episode)
        print("done.")

        self.policy = policy

    def LSTDQ_iteration(self, samples, policy, precondition_value=.0001, use_optimized=False):
        """
        Compute Q value function of current policy via LSTDQ iteration.
        If a matrix has full rank: scipy.linalg solver
        else: Least squares solver
        :param samples: data samples
        :param policy: policy to work with
        :param precondition_value: helps to ensure few samples give a matrix of full rank, choose 0 if not desired
        :param use_optimized: whether to use the "optimized" version to compute the next weights. Does not compute
        inverse of A, but it is slower due to a necessary loop.
        :return: the next weights of the policy
        """

        k = policy.basis_function.size()

        obs = samples[:, 0: self.ACTION_IDX]
        action_idx = samples[:, self.ACTION_IDX]
        reward = samples[:, self.REWARD_IDX: self.NEXT_OBS_IDX]
        next_obs = samples[:, self.NEXT_OBS_IDX: self.DONE_IDX]
        done = samples[:, self.DONE_IDX].astype(np.bool)

        phi = policy.basis_function(obs, action_idx)
        phi_next = np.zeros_like(phi)

        if np.any(~done):
            sel_next_obs = next_obs[~done]
            best_action = policy.get_best_action(sel_next_obs)
            phi_next[~done, :] = policy.basis_function(sel_next_obs, best_action)

        if not use_optimized:
            A = (phi.T @ (phi - self.gamma * phi_next) + np.identity(k) * precondition_value)
            b = (phi.T @ reward)

            # this is just to verify the matrix computations are correct and compared against a slower loop approach
            # a_mat, b_vec, phi_sa, phi_sprime = LSTDQ_iteration_validation(samples, precondition_value)

            rank_A = np.linalg.matrix_rank(A)

            if rank_A == k:
                w = scipy.linalg.solve(A, b)
            else:
                # print(f'A matrix does not have full rank {k} > {rank_A}. Using least squares solver.')
                w = scipy.linalg.lstsq(A, b)[0]

        else:
            # B is the approximate inverse of the A matrix from above
            # This avoids computing the inverse of A, but introduces a necessary loop,
            # because each updated is dependend on the previous B.
            # Therefor, we do not reconded using this.
            B = (1 / precondition_value) * np.identity(k)
            b = 0
            for i in range(len(phi)):
                p = phi[i].reshape(-1, 1)
                pn = phi_next[i].reshape(-1, 1)

                top = B @ (p @ (p - self.gamma * pn).T) @ B
                bottom = 1 + (p - self.gamma * pn).T @ B @ p
                B -= top / bottom
                b += p @ reward[i]

            w = B @ b

        return w.reshape((-1,))

    def LSTDQ_iteration_validation(self, samples, precondition_value):
        """
        loopy version of the above matrix LSTDQ to check for correctness of computation.
        :param samples: data samples
        :param precondition_value: Helps to ensure few samples give a matrix of full rank, choose 0 if not desired
        :return: a_mat, b_vec, phi, phi_next
        """
        k = self.policy.basis_function.size()

        a_mat = np.zeros((k, k))
        np.fill_diagonal(a_mat, precondition_value)

        b_vec = np.zeros((k, 1))

        phi = []
        phi_next = []

        for sample in samples:

            obs = sample[0: self.ACTION_IDX]
            action_idx = sample[self.ACTION_IDX]
            reward = sample[self.REWARD_IDX: self.NEXT_OBS_IDX]
            next_obs = sample[self.NEXT_OBS_IDX: self.DONE_IDX]
            done = sample[self.DONE_IDX].astype(np.bool)

            phi_sa = (self.policy.basis_function.evaluate(obs, action_idx).reshape((-1, 1)))

            if not done:
                best_action = self.policy.best_action(next_obs)
                phi_sprime = (self.policy.basis_function.evaluate(next_obs, best_action).reshape((-1, 1)))
            else:
                phi_sprime = np.zeros((k, 1))

            phi.append(phi_sa)
            phi_next.append(phi_sprime)

            a_mat += phi_sa.dot((phi_sa - self.gamma * phi_sprime).T)
            b_vec += phi_sa * reward

        return a_mat, b_vec, np.array(phi), np.array(phi_next)

    def train(self, policy_step_episodes=3, do_render=True, max_policy_steps=100):
        """
        Execute LSTDQ_iteration multiple times and display intermediate results, if wanted.

        :param policy_step_episodes: the number of episodes which are simulated between policy update steps
        :param do_render: whethter to render the simulated episodes
        :param max_policy_steps: the maximum number of policy update steps
        """

        delta = np.inf
        episodes = 0
        total_steps = 0

        episode_reward = 0
        episode_steps = 0

        print("Starting training")
        policy_step = 0

        done = False
        obs = self.env.reset()
        while delta >= self.theta:
            if policy_step_episodes > 0:
                total_steps += 1
                episode_steps += 1

                if self.normalize:
                    obs = normalize_state(self.env, obs, low=self.low, high=self.high)

                action_idx = self.policy.choose_action(obs)
                action = self.discrete_actions[action_idx]

                next_obs, reward, done, _ = self.env.step(action)

                if do_render and episode_steps % 12 == 0:
                    self.env.render()

                episode_reward += reward

                obs = next_obs

                if done:
                    obs = self.env.reset()
                    episodes += 1
                    print(
                        "Episode {:5d} -- total steps: {:8d} > avg reward: {:.10f} -- episode steps: {:4d} "
                        "-- episode reward: {:5.5f} -- delta: {:6.10f}".format(episodes, total_steps,
                                episode_reward / episode_steps, episode_steps, episode_reward, delta))

                    episode_steps = 0
                    episode_reward = 0

            if policy_step_episodes == 0 or (done and episodes % policy_step_episodes == 0):
                # update the current policy
                new_weights = self.LSTDQ_iteration(self.memory.memory, self.policy)
                delta = np.linalg.norm(new_weights - self.policy.w)
                self.policy.w = new_weights

                print(f"Policy ({policy_step}) delta: {delta}")

                policy_step += 1

                if policy_step >= max_policy_steps:
                    print("Reached max policy update steps. Stopped training.")
                    break

        print("Finished training")
