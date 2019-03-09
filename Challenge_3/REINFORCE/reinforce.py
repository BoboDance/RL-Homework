import torch
import numpy as np

from Challenge_3 import Policy
from Challenge_3.Util import get_samples, get_returns_torch


class REINFORCE:
    def __init__(self, env, model_policy: Policy, gamma, lr,
                 normalize_observations=False, low=None, high=None, use_tensorboard=False):

        self.env = env
        self.model_policy = model_policy
        self.gamma = gamma
        self.normalize_observations = normalize_observations
        self.low = low
        self.high = high
        self.use_tensorboard = use_tensorboard

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

    def train(self, min_steps=20000, max_episodes=100, max_episode_steps=10000, render_episodes_mod: int = None,
              save_best: bool = True, save_path="../checkpoints/reinforce_best_weights.pth"):
        """
        Runs the full training loop over several episodes. The best model weights are saved each time progress was made.

        :param min_steps: The minimum amount of simulation steps before one update step
        :param max_episodes: The maximum number of training episodes
        :param max_episode_steps: The maximum amount of steps of a single episode
        :param render_episodes_mod: Number of episodes when a new run will be rendered
        :param save_best: Defines if the best model policy shall be saved during training
        :param save_path: The path where the best policy will be stored
        :return: The final episode reached after training
        """
        episode = 0
        total_steps = 0

        self.model_policy.train()

        # try-catch block to allow stopping the training process on KeyboardInterrupt
        try:
            while episode < max_episodes:

                sample_episodes, sample_steps, memory, episode_rewards, episode_steps = \
                    get_samples(self.env, self.model_policy, min_steps=min_steps,
                                normalize_observations=self.normalize_observations, low=self.low, high=self.high)

                episode += sample_episodes
                total_steps += sample_steps
                episode_reward = episode_rewards.mean()
                episode_steps = episode_steps.mean()

                # check if episode reward is better than best model so far
                if self.best_episode_reward is None or episode_reward > self.best_episode_reward:
                    self.best_episode_reward = episode_reward
                    print("New best model has reward {:5.5f}".format(self.best_episode_reward))

                    if save_best:
                        torch.save(self.model_policy.state_dict(), save_path)
                        print("Model saved.")

                # Backward pass: Calculate the return for each step in the simulated episode (backwards)
                returns = get_returns_torch(memory[:, 2], self.gamma, memory[:, 3])
                log_confidence = torch.stack(list(memory[:, 4]))
                episode_loss = -log_confidence * returns
                episode_loss = episode_loss.sum()

                # do the optimization step
                self.optimizer.zero_grad()
                episode_loss.backward()
                self.optimizer.step()

                # Output statistics and save the model if wanted
                print(
                    "\rEpisode {:5d} -- total steps: {:8d} > avg steps: {:4f} -- avg reward: {:5.5f} "
                    "-- training loss: {:10.5f}".format(episode, total_steps, episode_steps,
                                                        episode_reward, episode_loss.float()))

                if self.use_tensorboard:
                    self.writer.add_scalar("avg_steps", episode_steps, episode)
                    self.writer.add_scalar("avg_reward", episode_reward, episode)
                    self.writer.add_scalar("loss", episode_loss, total_steps)


        except KeyboardInterrupt:
            print("Interrupted.")
        except:
            raise

        return episode
