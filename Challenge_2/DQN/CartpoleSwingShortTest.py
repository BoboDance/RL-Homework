import torch
import numpy as np
import gym
from torch.optim.lr_scheduler import StepLR

from Challenge_2.DQN.DQN import DQN
from Challenge_2.DQN.DQNModel import DQNModel
from Challenge_2.DQN.Util import evaluate
from challenge1 import load_model

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make("CartpoleSwingShort-v0")
env.seed(seed)

discrete_actions = np.array([-20, 20])

Q = DQNModel(env, discrete_actions, optimizer="adam", lr=1e-3)

dqn = DQN(env, Q, 20000, 1000, 42, 100, 0.7, 1, 0.15, 10000, 300, 5000, lr_scheduler=StepLR(Q.optimizer, 5000, 0.9))
trained_episodes = dqn.train()
# load_model(env, Q, "./checkpoints/Q_Pendulum-v0_100_-133.66.pth.tar")

eval_reward_mean = evaluate(env, Q, episodes=100, render=5)
# save_model(env, Q, trained_episodes, eval_reward_mean)

env.close()
