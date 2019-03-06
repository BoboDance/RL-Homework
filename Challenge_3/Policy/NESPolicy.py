import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from Challenge_3.Util import init_weights


class NESPolicy(nn.Module):
    def __init__(self, env, n_hidden_units=16):
        """
        Create a continuous policy which samples its actions using predicted a gaussian for a given observation.

        :param env: Gym environment object for which this policy object is created
        :param n_hidden_units: Number of nodes in the hidden layer
        :param state_dependent_sigma: Predict sigma together with the mean or use the same sigma for every observation
        """

        super(NESPolicy, self).__init__()

        self.n_inputs = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.action_space = env.action_space

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, n_hidden_units),
            nn.ReLU(),
            nn.Linear(n_hidden_units, self.n_actions)
        )

        self.apply(init_weights)

        self.train()

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: means and sigma_squared for each action
        """
        return self.model(inputs.float())
