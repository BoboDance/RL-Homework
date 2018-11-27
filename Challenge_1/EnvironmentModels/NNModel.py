import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


class NNModel(torch.nn.Module):

    def __init__(self, env, lr=1e-3):
        super(NNModel, self).__init__()

        self.env = env

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.high_state = self.env.observation_space.high
        self.high_action = self.env.action_space.high

        self.low_state = self.env.observation_space.low
        self.low_action = self.env.action_space.low

        self.n_inputs = self.state_dim + self.action_dim
        self.n_outputs = self.state_dim

        # network architecture specification
        fc1_out = 200
        fc2_out = 100

        self.fc1 = nn.Linear(self.n_inputs, fc1_out)
        self.fc2 = nn.Linear(self.n_inputs, fc2_out)

        # * reward head
        self.reward = nn.Linear(fc2_out, 1)

        # * dynamics head
        self.s_prime = nn.Linear(fc1_out, self.n_outputs)

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

        x1 = self.fc2(inputs)
        x1 = F.relu(x1)
        reward = self.reward(x1)

        x2 = self.fc1(inputs)
        x2 = F.relu(x2)
        s_prime = torch.from_numpy(self.high_state) * torch.tanh(self.s_prime(x2))

        return reward, s_prime

    def train_network(self, s_a, s_prime, reward, steps):
        s_prime = torch.from_numpy(s_prime).float()
        s_prime = s_prime.view(-1, self.n_outputs)

        reward = torch.from_numpy(reward).float()
        reward = reward.view(-1, 1)

        s_a = torch.from_numpy(s_a).float()
        s_a = s_a.view(-1, self.n_inputs)

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

    def validate_model(self, s_a, s_prime, reward):

        self.eval()

        with torch.no_grad():
            s_prime = torch.from_numpy(s_prime).float()
            s_prime = s_prime.view(-1, self.n_outputs)

            reward = torch.from_numpy(reward).float()
            reward = reward.view(-1, 1)

            s_a = torch.from_numpy(s_a).float()
            s_a = s_a.view(-1, self.n_inputs)

            reward_out, s_prime_out = self(s_a)

            mse_dynamics_test = ((s_prime.item() - s_prime_out.item()) ** 2).mean(axis=0)
            mse_reward_test = ((reward.item() - reward_out.item()) ** 2).mean()

            print("Test MSE for dynamics: {}".format(mse_dynamics_test))
            print("Test MSE for reward: {}".format(mse_reward_test))

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()
