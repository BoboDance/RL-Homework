import torch.nn as nn
from Challenge_2.DQN.Models.DQNModel import DQNModel
import gym
import numpy as np


class DQNSwingShortModel(DQNModel):

    def __init__(self, env: gym.Env, discrete_actions: np.ndarray, lr: float = 1e-3, optimizer: str = 'adam'):
        """
        Constructor
        :param env: Gym environment object
        :param discrete_actions: Numpy array which stores the values for the action bins
        :param lr: Learning rate to use
        :param optimizer: Optimizer to use
        """
        super(DQNSwingShortModel, self).__init__(env, discrete_actions, lr, optimizer)

    def get_model(self) -> nn.Sequential:
        """
        Defines the model architecture for the swing-up environment
        :return: Pytorch sequential model
        """
        hidden = 30
        act = nn.ReLU()

        return nn.Sequential(
            nn.Linear(self.n_inputs, hidden),
            act,
            nn.Linear(hidden, self.n_outputs),
        )
