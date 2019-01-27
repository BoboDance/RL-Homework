import torch.nn as nn
from Challenge_2.DQN.Models.DQNModel import DQNModel


class DQNSwingShortModel(DQNModel):

    def __init__(self, env, discrete_actions, scaling=None, lr=1e-3, optimizer='adam'):
        super(DQNSwingShortModel, self).__init__(env, discrete_actions, scaling, lr, optimizer)

    def get_model(self):
        hidden = 30
        act = nn.ReLU()

        return nn.Sequential(
            nn.Linear(self.n_inputs, hidden),
            act,
            nn.Linear(hidden, self.n_outputs),
        )