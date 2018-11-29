import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


class NNModel(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, scaling=None, lr=1e-3):
        super(NNModel, self).__init__()

        self.scaling = scaling
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

        # self.state_dim = self.env.observation_space.shape[0]
        # self.action_dim = self.env.action_space.shape[0]
        #
        # self.high_state = self.env.observation_space.high
        # self.high_action = self.env.action_space.high
        #
        # self.low_state = self.env.observation_space.low
        # self.low_action = self.env.action_space.low

        # self.n_inputs = self.state_dim + self.action_dim
        # self.n_outputs = self.state_dim

        # network architecture specification
        hidden = 100

        self.fc1 = nn.Linear(self.n_inputs, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, self.n_outputs)

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

        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        if self.scaling is not None:
            out = torch.from_numpy(self.scaling) * torch.tanh(self.out(x))
        else:
            out = self.out(x)

        return out

    def train_network(self, X, y, steps, path):

        y = torch.from_numpy(y).float()
        X = torch.from_numpy(X).float()

        criterion = nn.MSELoss()

        for i in range(steps):
            self.optimizer.zero_grad()

            out = self(X)

            loss = criterion(out, y)

            print("Step: {:d} -- total loss: {:3.8f}".format(i, loss.item()))

            loss.backward()

            self.optimizer.step()

        torch.save(self.state_dict(), path)

    def validate_model(self, X, y):

        self.eval()

        with torch.no_grad():
            # y = torch.from_numpy(y).float()
            X = torch.from_numpy(X).float()

            out = self(X)

            mse_test = ((out.detach().numpy() - y) ** 2).mean(axis=0)

            print("Test MSE: {}".format(mse_test))

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def predict(self, X):
        return self(X)
