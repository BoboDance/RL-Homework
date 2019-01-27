import torch
import numpy as np
import gym
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from Challenge_2.Common.Util import evaluate
from Challenge_2.DQN.DQN import DQN
from Challenge_2.DQN.DQNSwingShortModel import DQNSwingShortModel
from Challenge_2.DQN.Util import get_policy_fun, save_model
from torch import nn

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make("CartpoleSwingShort-v0")
env.seed(seed)

# Golden Hyperparameters
min_action = -5
max_action = 5
nb_bins = 9
lr = 1e-4
memory_size = int(1e6)
gamma = 0.99
eps_start = 1
eps_end = 0.01
eps_decay = 1e5  # they divided epsilon by 1.01 after every episode
max_episodes = 300
max_episode_length = 1e4
minibatch_size = 1000
target_model_update_steps = 3000
optimizer = "adam"
lr_scheduler = None
loss = nn.SmoothL1Loss()
normalize = False
anti_sucide = False
edge_fear_threshold = 0.3
use_tensorboard = False

discrete_actions = np.linspace(min_action, max_action, nb_bins)

Q = DQNSwingShortModel(env, discrete_actions, optimizer=optimizer, lr=lr)

# test for dividing lr by 2 every 50 episodes - start_lr = 1e-3 - noramlize=False
# Stats: Episodes 100, avg: 932.2933493737411, std: 22.219861482005673
# lr_scheduler = StepLR(Q.optimizer, max_episode_length*50, 0.5)  # None
# lr_scheduler = CosineAnnealingLR(Q.optimizer, 10) #T_max=max_episode_length)

dqn = DQN(env, Q, memory_size=memory_size, initial_memory_count=minibatch_size, minibatch_size=minibatch_size,
          target_model_update_steps=target_model_update_steps, gamma=gamma,
          eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, max_episodes=max_episodes,
          max_steps_per_episode=max_episode_length, lr_scheduler=lr_scheduler, loss=loss, normalize=normalize,
          anti_suicide=anti_suicide, use_tensorboard=use_tensorboard)

trained_episodes = dqn.train()
# load_model(env, Q, "./checkpoints/Q_Pendulum-v0_100_-133.66.pth.tar")

# load the best weights again
Q.load_state_dict(torch.load("./checkpoints/best_weights.pth"))

eval_reward_mean = evaluate(env, get_policy_fun(Q), episodes=100, render=3)
#save_model(env, Q, trained_episodes, eval_reward_mean)

env.close()
