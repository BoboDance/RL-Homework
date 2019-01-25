import torch
import numpy as np
import gym

from Challenge_2.DQN.DQN import DQN
from Challenge_2.DQN.DQNModel import DQNModel
from Challenge_2.DQN.Util import evaluate

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make("Pendulum-v0")
env.seed(seed)

discrete_actions = np.array([-2, 2])

Q = DQNModel(env, discrete_actions, optimizer="adam", lr=1e-3)

dqn = DQN(env, Q, 10000, 2000, 128, 30, 0.9999, 1, 0.1, 5000, 100, 5000)
trained_episodes = dqn.train()
# load_model(env, Q, "./checkpoints/Q_Pendulum-v0_100_-133.66.pth.tar")

eval_reward_mean = evaluate(env, Q, episodes=50, render=1)
# save_model(env, Q, trained_episodes, eval_reward_mean)

env.close()