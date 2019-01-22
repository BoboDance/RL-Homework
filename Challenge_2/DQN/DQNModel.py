import torch
import torch.nn as nn
from torch import optim
import numpy as np


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


class DQNModel(torch.nn.Module):

    def __init__(self, env, discrete_actions, scaling=None, lr=1e-3, optimizer='adam'):
        super(DQNModel, self).__init__()

        self.scaling = scaling
        self.discrete_actions = discrete_actions
        self.n_outputs = len(discrete_actions)
        self.n_inputs = env.observation_space.shape[0]

        # network architecture specification
        hidden = 15

        act = nn.ReLU()

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, hidden),
            act,
            nn.Linear(hidden, hidden),
            act,
            nn.Linear(hidden, hidden),
            act,
            nn.Linear(hidden, self.n_outputs),
        )

        # initialize the weights
        self.apply(init_weights)
        self.train()

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = None

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: reward, mu, sigma
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs)

        inputs = inputs.float()

        out = self.model(inputs)

        if self.scaling is not None:
            out = torch.from_numpy(self.scaling).float() * torch.tanh(out)

        return out.float()

    def predict(self, X):
        self.eval()
        X = torch.from_numpy(X)
        with torch.no_grad():
            return self(X).detach().numpy()
