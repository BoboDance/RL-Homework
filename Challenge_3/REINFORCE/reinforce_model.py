"""
@file: reinforce_model
Created on 20.02.19
@project: RL-Homework
@author: queensgambit

Please describe what the content of this file is about
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def init_weights(m):
    """
    Initializes the weights of the model by using kaiming normal initialization.
    :param m: Handle for the model
    :return:
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


class REINFORCEModel(nn.Module):
    def __init__(self, env,  n_actions, n_hidden_units=16):
        """
        Constructor

        :param env: Gym environment object
        :param n_actions: Number of discrete actions (will specify output of the network)
        :param n_hidden_units: Number of nodes in the hidden layer
        """

        super(REINFORCEModel, self).__init__()

        self.n_inputs = env.observation_space.shape[0]
        self.n_outputs = n_actions
        self.n_hidden_units = n_hidden_units

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden_units),
            nn.ReLU(),
            nn.Linear(self.n_hidden_units, self.n_outputs),
            nn.Softmax(1)
        )

        # initialize the weights
        self.apply(init_weights)

        self.train()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: Softmax values for each action
        """
        out = self.model(inputs)
        return out.float()

    def choose_action_by_sampling(self, observation):
        """
        Calls
        :param observation:
        :return: action.item(): Class idx which was chosen
                 distribution.log_prob(action): Confidence for choosing it
        """
        # convert the given observation into a torch tensor
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        # get the probability distribution from the neural net
        probabilities = self.forward(observation).cpu()
        # define a categorical distribution and sample from it
        distribution = Categorical(probabilities)
        action = distribution.sample()

        return action.item(), distribution.log_prob(action)
