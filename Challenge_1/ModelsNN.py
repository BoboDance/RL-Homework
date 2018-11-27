import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


class NNModel(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, high_state, lr=1e-3):
        super(NNModel, self).__init__()

        self.high_state = high_state
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        # network architecture specification
        fc1_out = 100
        fc2_out = 100

        self.fc1 = nn.Linear(n_inputs, fc1_out)
        self.fc2 = nn.Linear(n_inputs, fc2_out)

        # Define the two heads of the network
        # -----------------------------------

        # * reward head
        self.reward_linear = nn.Linear(fc2_out, 1)

        # * dynamics head
        self.out = nn.Linear(fc1_out, n_outputs)
        # self.sigma = nn.Linear(fc1_out, self.state_dim)

        # initialize the weights
        self.apply(init_weights)
        self.train()

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: reward, mu, sigma
        """
        inputs = inputs.float()

        x = self.fc1(inputs)
        x = F.relu(x)

        x1 = self.fc2(inputs)
        x1 = F.relu(x1)

        reward = self.reward_linear(x1)
        out = torch.from_numpy(self.high_state) * torch.tanh(self.out(x))
        return reward, out

    def train_network(self, s_a, s_prime, reward, steps):
        s_prime = torch.from_numpy(s_prime).float()
        s_prime = s_prime.view(-1, 2)

        reward = torch.from_numpy(reward).float()
        reward = reward.view(-1, 1)

        s_a = torch.from_numpy(s_a).float()
        s_a = s_a.view(-1, 3)

        criterion = nn.MSELoss()

        for i in range(steps):
            self.optimizer.zero_grad()

            reward_out, s_prime_out = self(s_a)

            state_loss = criterion(s_prime_out, s_prime)
            reward_loss = criterion(reward_out, reward)
            loss = state_loss + reward_loss

            print("Step: {:d} -- state loss: {:3.8f} -- reward loss: {:3.8f} "
                  "-- total loss: {:3.8f}".format(i, state_loss.item(), reward_loss.item(), loss.item()))

            loss.backward()

            self.optimizer.step()

        torch.save(self.state_dict(), "./NN-state_dict")

