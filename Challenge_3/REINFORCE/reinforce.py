"""
@file: reinforce
Created on 20.02.19
@project: RL-Homework
@author: queensgambit

Please describe what the content of this file is about
"""
from collections import deque

from Challenge_3.REINFORCE.reinforce_model import REINFORCEModel
import torch
import numpy as np


class REINFORCE:

    def __init__(self, env, model_policy: REINFORCEModel, discrete_actions, gamma, lr,
                 normalize=False, low=None, high=None, use_tensorboard=False,
                 save_path="./checkpoints/best_weights.pth"):

            self.env = env
            self.model_policy = model_policy
            self.discrete_actions = discrete_actions
            self.gamma = gamma
            self.normalize = normalize
            self.low = low
            self.high = high
            self.use_tensorboard = use_tensorboard
            self.save_path = save_path

            self.saved_log_probs = []

            self.rewards = []
            self.optimizer = torch.optim.Adam(model_policy.parameters(), lr=lr)
            self.eps = np.finfo(np.float32).eps.item()

            # save the current best episode reward
            self.best_episode_reward = None

            if self.use_tensorboard:
                from tensorboardX import SummaryWriter

                # init tensorboard writer for better visualizations
                self.writer = SummaryWriter()

            self.dim_obs = env.observation_space.shape[0]
            self.dim_action = env.action_space.shape[0]

            print("Used discrete actions: ", self.discrete_actions)

    def train(self, max_episodes = 100, max_episode_steps = 10000, render_episodes_mod: int = None, save_best: bool = True):
        """
        Runs the full training loop over several episodes. The best model weights are saved each time progress was made.

        :param max_episodes: The maximum number of training episodes.
        :param max_episode_steps: The maximum amount of steps of a single episode.
        :param render_episodes_mod: Number of episodes when a new run will be rendered
        :param save_best: Defines if the best model policy shall be saved during training.
                          Saved in checkpoints/best_weights.pth
        :return: The final episode reached after training
        """
        episode = 0
        total_steps = 0

        # try-catch block to allow stopping the training process on KeyboardInterrupt
        try:
            while episode < max_episodes:

                state = self.env.reset()
                episode_reward = 0
                episode_steps = 0
                self.saved_log_probs = []
                self.rewards = []

                # Forward pass: Simulate one episode and save its stats
                for episode_steps in range(0, max_episode_steps):
                    # Choose an action and remember its log probability
                    action_idx, log_prob = self.model_policy.choose_action_by_sampling(state)
                    action = np.array([self.discrete_actions[action_idx]])
                    self.saved_log_probs.append(log_prob)

                    # Make a step in the environment
                    state, reward, done, _ = self.env.step(action)
                    if render_episodes_mod is not None and episode > 0 and episode % render_episodes_mod == 0:
                       self.env.render()
                    self.rewards.append(reward)

                    episode_reward += reward
                    total_steps += 1

                    if done:
                        break

                episode += 1
                avg_reward = episode_reward / episode_steps

                # Backward pass: Calculate the return for each step in the simulated episode (backwards)
                R = 0
                episode_loss = []
                returns = []
                for r in self.rewards[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns)
                # Normalize the returns
                returns = (returns - returns.mean()) / (returns.std() + self.eps)
                # Get our loss over the whole trajectory
                for log_prob, R in zip(self.saved_log_probs, returns):
                    episode_loss.append(-log_prob * R)
                episode_loss = torch.cat(episode_loss).sum()

                # do the optimization step
                self.optimizer.zero_grad()
                episode_loss.backward()
                self.optimizer.step()

                # Output statistics and save the model if wanted
                print(
                    "\rEpisode {:5d} -- total steps: {:8d} > avg reward: {:.10f} -- steps: {:4d} -- reward: {:5.5f} "
                    "-- training loss: {:10.5f}".format(episode, total_steps, avg_reward, episode_steps,
                                                        episode_reward, episode_loss.float()))

                if self.use_tensorboard:
                    self.writer.add_scalar("avg_reward", avg_reward, episode)
                    self.writer.add_scalar("total_reward", episode_reward, episode)
                    self.writer.add_scalar("loss", episode_loss, total_steps)

                # check if episode reward is better than best model so far
                if (self.best_episode_reward is None or episode_reward > self.best_episode_reward):
                    self.best_episode_reward = episode_reward
                    print("New best model has reward {:5.5f}".format(self.best_episode_reward))

                    if save_best:
                        torch.save(self.model_policy.state_dict(), self.save_path)
                        print("Model saved.")

        except(KeyboardInterrupt):
            print("Interrupted.")
        except:
            raise

        return episode
