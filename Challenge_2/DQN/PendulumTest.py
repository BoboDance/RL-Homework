import torch
import numpy as np
import gym

from Challenge_2.Common.Util import evaluate
from Challenge_2.DQN.DQN import DQN
from Challenge_2.DQN.DQNPendulumModel import DQNPendulumModel
from Challenge_2.DQN.Util import get_policy_fun

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

env = gym.make("Pendulum-v0")
env.seed(seed)

discrete_actions = np.array([-2, 2])

Q = DQNPendulumModel(env, discrete_actions, optimizer="adam", lr=1e-3)

dqn = DQN(env, Q, 10000, 2000, 128, 30, 0.9999, 1, 0.1, 5000, 100, 5000)
trained_episodes = dqn.train(save_best=False)
# load_model(env, Q, "./checkpoints/Q_Pendulum-v0_100_-133.66.pth.tar")

eval_reward_mean = evaluate(env, get_policy_fun(env, Q, False), episodes=50, render=1)
# save_model(env, Q, trained_episodes, eval_reward_mean)

env.close()
