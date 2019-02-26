import torch
import numpy as np

from Challenge_3 import Policy
from Challenge_3.Util import normalize_state


class REINFORCE:
    def __init__(self, env, model_policy: Policy, gamma, lr,
                 normalize_observations=False, low=None, high=None, use_tensorboard=False,
                 save_path="./checkpoints/best_weights.pth"):

        self.env = env
        self.model_policy = model_policy
        self.gamma = gamma
        self.normalize_observations = normalize_observations
        self.low = low
        self.high = high
        self.use_tensorboard = use_tensorboard
        self.save_path = save_path

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

    def train(self, max_episodes=100, max_episode_steps=10000, render_episodes_mod: int = None, save_best: bool = True):
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
                if self.normalize_observations:
                    state = normalize_state(self.env, state, low=self.low, high=self.high)

                episode_reward = 0
                episode_steps = 0

                saved_log_probs = []
                # saved_log_prob_derivations = []
                saved_rewards = []

                # Forward pass: Simulate one episode and save its stats
                for episode_steps in range(0, max_episode_steps):
                    # Choose an action and remember its log probability
                    action, log_prob = self.model_policy.choose_action_by_sampling(state)

                    # Make a step in the environment
                    state, reward, done, _ = self.env.step(action)
                    if self.normalize_observations:
                        state = normalize_state(self.env, state, low=self.low, high=self.high)

                    # Hotfix because Levitation-v1 returns numpy array instead of single value
                    if type(reward) is np.ndarray:
                        reward = reward[0]

                    # Render the environment if we want to
                    if render_episodes_mod is not None and episode > 0 and episode % render_episodes_mod == 0:
                        self.env.render()

                    # Store the log probability and the reward of the current step for the backward pass later
                    saved_log_probs.append(log_prob)
                    saved_rewards.append(reward)

                    # also get the derivative of the model parameters
                    # self.optimizer.zero_grad()
                    # log_prob.backward(retain_graph=True)
                    # gradients = []
                    # for param in self.model_policy.parameters():
                    #     gradients.extend(param.grad.detach().numpy().flatten())
                    # saved_log_prob_derivations.append(np.array(gradients))

                    episode_reward += reward
                    total_steps += 1

                    if done:
                        break

                episode += 1
                avg_reward = episode_reward / episode_steps

                # params = saved_log_prob_derivations[0].shape[0]
                # fisher_information = np.zeros((params, params))
                # for dev_log_pi in saved_log_prob_derivations:
                #     fisher_information += dev_log_pi @ dev_log_pi.T
                # fisher_information *= 1 / len(saved_log_prob_derivations)
                # print(fisher_information)

                # Backward pass: Calculate the return for each step in the simulated episode (backwards)
                R = 0
                episode_loss = []
                returns = []
                for r in saved_rewards[::-1]:
                    R = r + self.gamma * R
                    returns.insert(0, R)
                returns = torch.Tensor(returns)
                # Normalize the returns
                # returns = (returns - returns.mean()) / (returns.std() + self.eps)
                # Get our loss over the whole trajectory
                for log_prob, R in zip(saved_log_probs, returns):
                    episode_loss.append(-log_prob * R)
                episode_loss = torch.stack(episode_loss).mean()

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
                if self.best_episode_reward is None or episode_reward > self.best_episode_reward:
                    self.best_episode_reward = episode_reward
                    print("New best model has reward {:5.5f}".format(self.best_episode_reward))

                    if save_best:
                        torch.save(self.model_policy.state_dict(), self.save_path)
                        print("Model saved.")

        except KeyboardInterrupt:
            print("Interrupted.")
        except:
            raise

        return episode
