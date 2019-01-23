import torch
import numpy as np
import gym
from torch.optim.lr_scheduler import StepLR

from Challenge_2.DQN.DQN import DQN
from Challenge_2.DQN.DQNModel import DQNModel
from Challenge_2.DQN.Util import evaluate
from challenge1 import load_model
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
eps_decay = 1e5 # they divided epsilon by 1.01 after every episode
max_episodes = 300
max_episode_length = 2000
minibatch_size = 500
optimizer = "adam"
lr_scheduler = None  # StepLR(Q.optimizer, 5000, 0.9)
loss = nn.SmoothL1Loss()

discrete_actions = np.linspace(min_action, max_action, nb_bins)

Q = DQNModel(env, discrete_actions, optimizer="adam", lr=lr)

dqn = DQN(env, Q, memory_size=memory_size, initial_memory_count=minibatch_size, minibatch_size=minibatch_size,
          target_model_update_steps=minibatch_size, gamma=gamma,
          eps_start=eps_start, eps_end=eps_end, eps_decay=eps_decay, max_episodes=max_episodes,
          max_steps_per_episode=max_episode_length, lr_scheduler=lr_scheduler, loss=loss)

trained_episodes = dqn.train()
# load_model(env, Q, "./checkpoints/Q_Pendulum-v0_100_-133.66.pth.tar")

# load the best weights again
Q.load_state_dict(torch.load("./checkpoints/best_weights.pth"))

eval_reward_mean = evaluate(env, Q, episodes=100, render=5)
#save_model(env, Q, trained_episodes, eval_reward_mean)

env.close()
