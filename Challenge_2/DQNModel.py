import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


act = nn.ReLU()

class DQNModel(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, scaling=None, lr=1e-3, optimizer='adam'):
        super(DQNModel, self).__init__()

        self.scaling = scaling
        self.n_outputs = n_outputs
        self.n_inputs = n_inputs

        # network architecture specification
        hidden = 42

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, hidden),
            act,
            nn.Linear(hidden, hidden),
            act,
            nn.Dropout(.3),
            nn.Linear(hidden, hidden),
            act,
            nn.Dropout(.3),
            nn.Linear(hidden, self.n_outputs),
        )

        # initialize the weights
        self.apply(init_weights)
        self.train()

        #if optimizer == 'adam':
        #    self.optimizer = optim.Adam(self.parameters(), lr=lr)
        #elif optimizer == 'sgd':
        #    self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def copy_state_dict(self, other_model):
        self.load_state_dict(other_model.state_dict())

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: reward, mu, sigma
        """
        inputs = inputs.float()

        out = self.model(inputs)

        if self.scaling is not None:
            out = torch.from_numpy(self.scaling).float() * torch.tanh(out)

        return out

    def gradient_descend_step(self, X, y, loss_function = nn.SmoothL1Loss()):
        """
        Perform a gradient descend step on the given data.
        """

        self.train()

        y = torch.from_numpy(y).float()
        X = torch.from_numpy(X).float()

        self.optimizer.zero_grad()

        outputs = self(X)
        loss = loss_function(outputs, y)
        loss.backward()

        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()

        return loss.item()

    def get_best_action(self, observation, discrete_actions):
        """
        Get the action which returns the best predicted reward for the given observation.

        :param observation: the observation where the system is currently at
        :param discrete_actions: the discrete actions to consider
        :return: the action which is considered the best with the current model
        """

        values = self.get_action_values(observation, discrete_actions)
        # choose the action which gets the best value
        return np.array([discrete_actions[np.argmax(values)]])

    def get_best_value(self, observation, discrete_actions):
        values = self.get_action_values(observation, discrete_actions)
        return np.max(values)

    def get_action_values(self, observation, discrete_actions):
        # first build a matrix with duplicates of the last observation as rows for each discrete action
        tiled_observations = np.tile(observation, (len(discrete_actions), 1))
        # then append each discrete action value
        model_input = np.append(tiled_observations, discrete_actions.reshape(-1, 1), axis=-1)
        # predict all the values
        return self.predict(model_input)

    def predict(self, X):
        self.eval()
        X = torch.from_numpy(X)
        return self(X).detach().numpy()
