from collections import deque

import numpy as np
import torch
from Challenge_3.NaturalPG.npg_step import train_model, optimization_step
from Challenge_3.Util import normalize_state


class NaturalPG:
    def __init__(self, env, actor, gamma, critic=None, critic_lr = 0.005, normalize_observations=False, low=None, high=None, use_tensorboard=False):
        self.env = env

        self.dim_obs = env.observation_space.shape[0]
        self.dim_action = env.action_space.shape[0]

        self.normalize_observations = normalize_observations
        self.low = low
        self.high = high

        self.actor = actor
        self.critic = critic

        self.gamma = gamma

        if critic is not None:
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # save the current best episode reward
        self.best_episode_reward = None

        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter()

    def train(self, max_episodes=100, max_episode_steps=10000, render_episodes_mod: int = None):
        """
        Runs the full training loop over several episodes.

        :param max_episodes: The maximum number of training episodes.
        :param max_episode_steps: The maximum amount of steps of a single episode.
        :param render_episodes_mod: Number of episodes when a new run will be rendered
        :return: The final episode reached after training
        """
        episode = 0
        total_steps = 0

        try:
            for episode in range(max_episodes):
                self.actor.eval()
                if self.critic is not None:
                    self.critic.eval()
                memory = deque()

                state = self.env.reset()
                if self.normalize_observations:
                    state = normalize_state(self.env, state, low=self.low, high=self.high)
                episode_reward = 0
                episode_step = 0
                for episode_step in range(max_episode_steps):
                    action, log_confidence = self.actor.choose_action_by_sampling(state)

                    next_state, reward, done, _ = self.env.step(action)
                    if self.normalize_observations:
                        next_state = normalize_state(self.env, next_state, low=self.low, high=self.high)

                    if render_episodes_mod is not None and episode > 0 and episode % render_episodes_mod == 0:
                        self.env.render()

                    memory.append([state, action, reward, done, log_confidence])

                    episode_reward += reward
                    state = next_state

                    if done:
                        break

                total_steps += episode_step

                memory = np.array(memory)

                self.actor.train()
                if self.critic is None:
                    actor_loss, _ = optimization_step(self.actor, memory, self.gamma)
                    print(
                        "\rEpisode {:5d} -- total steps: {:8d} > steps: {:4d} -- reward: {:5.5f} "
                        "-- actor loss: {:.5f}".format(episode, total_steps, episode_step,
                                                       episode_reward, actor_loss))
                else:
                    self.critic.train()
                    actor_loss, critic_loss = \
                        optimization_step(self.actor, memory, self.gamma,
                                          critic=self.critic, optimizer_critic=self.optimizer_critic)
                    print(
                        "\rEpisode {:5d} -- total steps: {:8d} > steps: {:4d} -- reward: {:5.5f} "
                        "-- actor loss: {:.5f} -- critic loss: {:.5f}".format(episode, total_steps, episode_step,
                                                                              episode_reward, actor_loss, critic_loss))

                    if self.use_tensorboard:
                        self.writer.add_scalar("critic_loss", critic_loss, episode)

                if self.use_tensorboard:
                    self.writer.add_scalar("actor_loss", actor_loss, episode)
                    self.writer.add_scalar("total_reward", episode_reward, episode)
                    self.writer.add_scalar("steps", episode_step, episode)
                    self.writer.add_scalar("total_steps", total_steps, episode)

                episode += 1

        except KeyboardInterrupt:
            print("Interrupted.")
        except:
            raise

        return episode
