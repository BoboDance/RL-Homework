from collections import deque

import numpy as np
import torch
from Challenge_3.NPG.npg_step import train_model, optimization_step
from Challenge_3.Util import normalize_state, get_samples


class NaturalPG:
    def __init__(self, env, actor, gamma, min_steps, critic=None, critic_lr = 0.005, normalize_observations=False, low=None, high=None, use_tensorboard=False):
        self.env = env

        self.dim_obs = env.observation_space.shape[0]
        self.dim_action = env.action_space.shape[0]

        self.normalize_observations = normalize_observations
        self.low = low
        self.high = high

        self.actor = actor
        self.critic = critic

        self.gamma = gamma
        self.min_steps = min_steps

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
            while episode < max_episodes:
                self.actor.eval()
                if self.critic is not None:
                    self.critic.eval()

                sample_episodes, sample_steps, memory, episode_rewards, episode_steps = \
                    get_samples(self.env, self.actor, min_steps=self.min_steps,
                                normalize_observations=self.normalize_observations, low=self.low, high=self.high)

                if self.use_tensorboard:
                    tensorboard_total_steps = total_steps
                    for i in range(0, len(episode_rewards)):
                        tensorboard_episode = episode + i
                        reward = episode_rewards[i]
                        steps = episode_steps[i]
                        tensorboard_total_steps += total_steps

                        self.writer.add_scalar("episode_reward", reward, tensorboard_episode)
                        self.writer.add_scalar("episode_steps", steps, tensorboard_episode)
                        self.writer.add_scalar("total_steps", tensorboard_total_steps, tensorboard_episode)

                total_steps += sample_steps
                episode += sample_episodes

                self.actor.train()
                if self.critic is None:
                    actor_loss, _ = optimization_step(self.actor, memory, self.gamma)
                    print(
                        "\rEpisode {:5d} -- total steps: {:8d} > avg steps: {:.2f} -- avg reward: {:5.5f} "
                        "-- actor loss: {:.5f}".format(episode, total_steps, episode_steps.mean(),
                                                       episode_rewards.mean(), actor_loss))

                    if self.use_tensorboard:
                        self.writer.add_scalar("actor_loss", actor_loss, episode)


                else:
                    self.critic.train()
                    actor_loss, critic_loss = \
                        optimization_step(self.actor, memory, self.gamma,
                                          critic=self.critic, optimizer_critic=self.optimizer_critic)

                    print(
                        "\rEpisode {:5d} -- total steps: {:8d} > avg steps: {:.2f} -- avg reward: {:5.5f} "
                        "-- actor loss: {:.5f} -- critic loss: {:.5f}".format(episode, total_steps, episode_steps.mean(),
                                                       episode_rewards.mean(), actor_loss, critic_loss))

                    if self.use_tensorboard:
                        self.writer.add_scalar("actor_loss", actor_loss, episode)
                        self.writer.add_scalar("critic_loss", critic_loss, episode)

        except KeyboardInterrupt:
            print("Interrupted.")
        except:
            raise

        return episode
