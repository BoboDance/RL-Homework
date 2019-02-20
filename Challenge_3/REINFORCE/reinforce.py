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

    def __init__(self, env, model_policy: REINFORCEModel, discrete_actions,
                 gamma, lr, max_episodes,
                 normalize=False, low=None,
                 high=None, use_tensorboard=False,
                 save_path="./checkpoints/best_weights.pth"):

            self.env = env
            self.model_policy = model_policy
            self.discrete_actions = discrete_actions
            self.gamma = gamma
            self.max_episodes = max_episodes
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

    def train(self, render_episodes_mod: int = None, save_best: bool = True):
        """
        Runs the full training loop over several episodes. The best model weights are saved each time progress was made.
        :param render_episodes_mod: Number of episodes when a new run will be rendered
        :param save_best: Defines if the best model policy shall be saved during training.
        Saved in checkpoints/best_weights.pth
        :return: The final episode reached after training
        """
        """
        max_t = 10000
        scores_deque = deque(maxlen=1000)
        scores = []
        for i_episode in range(1, self.max_episodes + 1):
            saved_log_probs = []
            rewards = []
            state = self.env.reset()
            for t in range(max_t):
                # action, log_prob = policy.act(normalize_state(state))
                action, log_prob = self.model_policy.choose_action_by_sampling(state)
                saved_log_probs.append(log_prob)
                action_continious = np.array([self.discrete_actions[action]])
                # print(action)
                # actions_taken.append(action_continious[0])
                # action = np.array([action])
                state, reward, done, _ = self.env.step(action_continious)
                # state, reward, done, _ = env.step(action)

                rewards.append(reward)
                if done:
                    # print('break:', t)
                    break
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            discounts = [self.gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a, b in zip(discounts, rewards)])
            # R = R[0]
            policy_loss = []
            for log_prob in saved_log_probs:
                # print(type(log_prob))
                # print(R)

                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()

            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

            if i_episode % render_episodes_mod == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            # if np.mean(scores_deque) >= 195.0:
            #    print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
            #                                                                               np.mean(scores_deque)))
            #    break

            return
        """
        episode = 0
        total_steps = 0

        # try-catch block to allow stopping the training process on KeyboardInterrupt
        try:
            while episode < self.max_episodes:

                state = self.env.reset()
                episode_reward = 0
                self.saved_log_probs = []
                self.rewards = []

                episode_steps = 0

                for episode_steps in range(1, 10000):  # Don't infinite loop while learning
                    action_idx, log_prob = self.model_policy.choose_action_by_sampling(state)
                    self.saved_log_probs.append(log_prob)
                    action = np.array([self.discrete_actions[action_idx]])
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

                R = 0
                episode_loss = []
                returns = []
                for r in self.rewards[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.tensor(returns)
                returns = (returns - returns.mean()) / (returns.std() + self.eps)
                for log_prob, R in zip(self.saved_log_probs, returns):
                    episode_loss.append(-log_prob * R)
                episode_loss = torch.cat(episode_loss).sum()

                if self.use_tensorboard:
                    self.writer.add_scalar("avg_reward", avg_reward, episode)
                    self.writer.add_scalar("total_reward", episode_reward, episode)

                # check if episode reward is better than best model so far
                if save_best and (self.best_episode_reward is None or episode_reward > self.best_episode_reward):
                    self.best_episode_reward = episode_reward
                    print("new best model with reward {:5.5f}".format(self.best_episode_reward))
                    torch.save(self.model_policy.state_dict(), self.save_path)

                # optimize
                self.optimizer.zero_grad()
                episode_loss.backward()
                self.optimizer.step()

                if episode % render_episodes_mod == 0:
                    print(
                        "\rEpisode {:5d} -- total steps: {:8d} > avg reward: {:.10f} -- steps: {:4d} -- reward: {:5.5f} "
                        "-- training loss: {:10.5f}".format(episode, total_steps, avg_reward, episode_steps,
                                                            episode_reward, episode_loss.float()))

                if self.use_tensorboard:
                    self.writer.add_scalar("loss", episode_loss, total_steps)

        except(KeyboardInterrupt):
            print("Interrupted.")
        except:
            raise

        return episode

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]
