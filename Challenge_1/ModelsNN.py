import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


class Dynamics(torch.nn.Module):

    def __init__(self, n_inputs, observation_space):
        super(Dynamics, self).__init__()

        self.observation_space = observation_space
        self.state_dim = observation_space.shape[0]

        # network architecture specification
        fc1_out = 200
        fc2_out = 200

        self.fc1 = nn.Linear(n_inputs, fc1_out)
        self.fc2 = nn.Linear(n_inputs, fc2_out)

        # Define the two heads of the network
        # -----------------------------------

        # * reward head
        # The reward head has only 1 output
        self.reward_linear = nn.Linear(fc2_out, 1)

        # * dynamics head
        # in the continuous case it has one output for the mean and one for the cov of the state dist
        # later the workers can sample from a normal distribution
        self.mu = nn.Linear(fc1_out, self.state_dim)
        self.sigma = nn.Linear(fc1_out, self.state_dim)

        # initialize the weights
        self.apply(init_weights)
        self.train()

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: reward, mu, sigma
        """
        inputs = inputs.float()
        x = self.fc1(inputs)
        x = F.relu6(x)
        x = self.fc2(x)
        x = F.relu6(x)

        # clip state space
        print(self.observation_space.high)
        print(self.observation_space.low)

        reward = self.reward_linear(x)
        mu = torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 1e-5
        return reward, mu, sigma
