import math

import numpy as np
import torch
import torch.nn as nn
import copy

from Challenge_2.Common.ReplayMemory import ReplayMemory
from Challenge_2.Common.Util import create_initial_samples, normalize_state
from Challenge_2.DQN.Models.DQNModel import DQNModel
from Challenge_2.DQN.Util import get_best_values, get_best_action, get_current_lr


class DQN(object):

    def __init__(self, env, Q: DQNModel, memory_size, initial_memory_count, minibatch_size,
                 target_model_update_steps, gamma, eps_start, eps_end, eps_decay, max_episodes,
                 max_steps_per_episode, lr_scheduler=None, loss=nn.SmoothL1Loss(), normalize=False, low=None,
                 high=None, anti_suicide=False, edge_fear_threshold=0.3, use_tensorboard=False, full_episode=False,
                 save_path="./checkpoints/best_weights.pth"):
        """
        Initializes the DQN wrapper.

        :param env: the environment to train on
        :param Q: the Q model for this environment
        :param memory_size: the size of the replay memory
        :param initial_memory_count: the amount of samples which will be initially created in the replay memory
        :param minibatch_size: the size of the minibatch used in the optimization step
        :param target_model_update_steps: the number of steps after which the target Q model is updated
        :param gamma: the discount factor
        :param eps_start: the value of eps at the beginning (exploration ~ eps)
        :param eps_end: the value of eps at the end
        :param eps_decay: defines how fast eps translates from eps_start to eps_end
        :param max_episodes: the maximum number of episodes used in training
        :param max_steps_per_episode: the maximum number of steps in each episode during training
        :param lr_scheduler: the learning rate scheduler
        :param loss: the loss used for the optimization step
        :param normalize: boolean which enables or disables state normalization into feature range of [0,1]
        :param low: manual upper limit for the observation space
        :param high: manual lower limit for the observation space
        :param anti_suicide: technique which applies reward shaping to avoid that the agent crashes against the wall
        :param edge_fear_threshold: threshold when the anti-sucicide technique is triggered.
                Only effective is anti-sucide is True
        :param use_tensorboard: Boolean which enables the usage of tensorboard logging
        :param full_episode: whether the initial sampling should do a full episode (needed for monitor to work..)
        :param save_path: path to save the best model to
        """

        self.env = env
        self.discrete_actions = Q.discrete_actions
        self.minibatch_size = minibatch_size
        self.Q = Q
        self.target_model_update_steps = target_model_update_steps
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.lr_scheduler = lr_scheduler
        self.loss = loss
        self.normalize = normalize
        self.low = low
        self.high = high
        self.anti_suicide = anti_suicide
        self.edge_fear_threshold = edge_fear_threshold
        self.use_tensorboard = use_tensorboard
        self.save_path = save_path

        # save the current best episode reward
        self.best_episode_reward = None

        if self.use_tensorboard:
            from tensorboardX import SummaryWriter

            # init tensorboard writer for better visualizations
            self.writer = SummaryWriter()

        self.dim_obs = env.observation_space.shape[0]
        self.dim_action = env.action_space.shape[0]

        print("Used discrete actions: ", self.discrete_actions)

        # transition: observation, action, reward, next observation, done
        self.transition_size = self.dim_obs + self.dim_action + 1 + self.dim_obs + 1
        self.ACTION_INDEX = self.dim_obs
        self.REWARD_INDEX = self.dim_obs + self.dim_action
        self.NEXT_OBS_INDEX = self.dim_obs + self.dim_action + 1
        self.DONE_INDEX = -1

        self.memory = ReplayMemory(memory_size, self.transition_size)
        create_initial_samples(env, self.memory, initial_memory_count, self.discrete_actions,
                               normalize=normalize, low=low, high=high, full_episode=full_episode)

        # create the target Q network which will only be evaluated and not trained
        self.target_Q = copy.deepcopy(self.Q)
        self.target_Q.optimizer = None
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.target_Q.eval()

    def get_expected_values(self, transitions: np.ndarray, model: DQNModel) -> np.ndarray:
        """
        Return the expected value of a batch.

        :param transitions: Transition matrix. First dimension number of batches. 2nd dimension is size of replay memory
        :param model: Current NN model
        :return: expected value for given transitions
        """
        y = transitions[:, self.REWARD_INDEX].reshape(-1, 1)
        not_done_filter = ~transitions[:, self.DONE_INDEX].astype(bool)
        next_obs = transitions[not_done_filter, self.NEXT_OBS_INDEX:self.NEXT_OBS_INDEX + self.dim_obs]
        y[not_done_filter] += self.gamma * get_best_values(model, next_obs).reshape(-1, 1)

        return y

    def get_eps(self, total_steps: int) -> float:
        """
        Returns the current epsilon value which describes how many random actions are done.
        :param total_steps: Current amount of steps taken in total over several episodes.
        :return:
        """
        # compute decaying eps_threshold to encourage exploration at the beginning
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * total_steps / self.eps_decay)

    def choose_action_eps(self, observation: np.ndarray, total_steps: int) -> int:
        """
        Chooses an action either randomly or based on the q-value.
        :param observation: Current environment state
        :param total_steps: Current amount of steps taken in total over several episodes.
        :return: Index of the bin of the chosen action
        """
        eps_threshold = self.get_eps(total_steps)

        # choose epsilon-greedy action
        if np.random.uniform() <= eps_threshold:
            action_idx = np.random.choice(range(len(self.discrete_actions)), 1)
        else:
            # choose the best action
            if self.normalize:
                observation = normalize_state(self.env, observation, low=self.low, high=self.high)

            action_idx = get_best_action(self.Q, observation)

        return action_idx

    def optimize(self) -> float:
        """
        Runs the optimization step for updating the model weights
        :return: The loss value
        """

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        self.Q.train()

        minibatch = self.memory.sample(self.minibatch_size)

        # calculate our y based on the target model
        expected_values = self.get_expected_values(minibatch, self.target_Q)
        expected_values = torch.from_numpy(expected_values).float()

        obs = minibatch[:, :self.dim_obs]
        # obs = torch.from_numpy(obs).float()

        actions = minibatch[:, self.ACTION_INDEX:self.REWARD_INDEX]
        actions = torch.from_numpy(actions).long()

        # perform gradient descend regarding to expected values on the current model
        self.Q.optimizer.zero_grad()

        # convert observation from numpy into pytorch tensor format for fwd and bwd propagation
        obs = torch.from_numpy(obs)

        values = self.Q(obs).gather(1, actions)
        loss_val = self.loss(values, expected_values)
        loss_val.mean().backward()

        torch.nn.utils.clip_grad_norm_(self.Q.parameters(), 1)

        self.Q.optimizer.step()

        return loss_val.item()

    def train(self, render_episodes_mod: int = None, save_best: bool = True):
        """
        Runs the full training loop over several episodes. The best model weights are saved each time progress was made.
        :param render_episodes_mod: Number of episodes when a new run will be rendered
        :param save_best: Defines if the best model policy shall be saved during training.
        Saved in checkpoints/best_weights.pth
        :return: The final episode reached after training
        """
        episode = 0
        total_steps = 0

        episode_steps = 0
        episode_reward = 0
        episode_loss = 0

        obs = self.env.reset()
        # try-catch block to allow stopping the training process on KeyboardInterrupt
        try:
            while episode < self.max_episodes:
                self.Q.eval()

                action_idx = self.choose_action_eps(obs, total_steps)
                action = self.discrete_actions[action_idx]

                next_obs, reward, done, _ = self.env.step(action)

                episode_reward += reward

                if self.anti_suicide:
                    if np.abs(obs[0]) > self.edge_fear_threshold:
                        print("-> edge area")
                        reward = -np.abs(obs[0])

                # save the observed step in our replay memory
                self.memory.push((*obs, *action_idx, reward, *next_obs, done))

                # remember last observation
                obs = next_obs

                if render_episodes_mod is not None and episode > 0 and episode % render_episodes_mod == 0:
                    self.env.render()

                loss = self.optimize()
                episode_loss += loss

                if self.use_tensorboard:
                    self.writer.add_scalar("loss", loss, total_steps)

                total_steps += 1
                episode_steps += 1

                # copy the current model each target_model_update_steps steps
                if total_steps % self.target_model_update_steps == 0:
                    self.target_Q.load_state_dict(self.Q.state_dict())

                avg_reward = episode_reward / episode_steps

                if episode_steps % 100 == 0:
                    print("\rEpisode {:5d} -- total steps: {:8d} > steps: {:4d} (Running)"
                          .format(episode + 1, total_steps, episode_steps), end="")

                if done or episode_steps >= self.max_steps_per_episode:
                    obs = self.env.reset()
                    episode += 1

                    print(
                        "\rEpisode {:5d} -- total steps: {:8d} > avg reward: {:.10f} -- steps: {:4d} -- reward: {:5.5f} "
                        "-- training loss: {:10.5f} -- lr: {:0.8f} -- eps: {:0.8f}"
                            .format(episode, total_steps, avg_reward, episode_steps, episode_reward, episode_loss,
                                    get_current_lr(self.Q.optimizer), self.get_eps(total_steps)))

                    if self.use_tensorboard:
                        self.writer.add_scalar("avg_reward", avg_reward, episode)
                        self.writer.add_scalar("total_reward", episode_reward, episode)
                        self.writer.add_scalar("first_best_value", get_best_values(self.Q, np.atleast_2d(obs))[0],
                                               episode)

                    self.Q.eval()

                    # check if episode reward is better than best model so far
                    if save_best and (self.best_episode_reward is None or episode_reward > self.best_episode_reward):
                        self.best_episode_reward = episode_reward
                        print("new best model with reward {:5.5f}".format(self.best_episode_reward))
                        torch.save(self.Q.state_dict(), self.save_path)

                    episode_steps = 0
                    episode_loss = 0
                    episode_reward = 0

        except(KeyboardInterrupt):
            print("Interrupted.")
        except:
            raise

        return episode
