import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueModel(nn.Module):
    def __init__(self, env, n_hidden_units=16):
        """
        Create a model for the value function for the given env. Uses a neural network with a single hidden layer.

        :param env: The environment for which this value estimator is created.
        :param n_hidden_units: Number of nodes in the hidden layer.
        """
        super(ValueModel, self).__init__()

        self.n_inputs = env.observation_space.shape[0]
        self.n_hidden_units = n_hidden_units

        self.input_to_hidden = nn.Linear(self.n_inputs, self.n_hidden_units)
        self.hidden_to_value = nn.Linear(self.n_hidden_units, 1)

        # get small values in the beginning
        self.hidden_to_value.weight.data.mul_(0.1)
        self.hidden_to_value.bias.data.mul_(0.0)

    def forward(self, observations):
        values = self.hidden_to_value(F.relu(self.input_to_hidden(observations)))
        return values


def train_model(model: nn.Module, inputs, targets, optimizer, epochs=3, batch_size=64, criterion=torch.nn.MSELoss()):
    n = len(inputs)
    arr = np.arange(n)
    input_tensor = torch.Tensor(inputs)

    total_loss = 0

    for epoch in range(epochs):
        np.random.shuffle(arr)

        for i in range(n // batch_size):
            batch_index = arr[batch_size * i: batch_size * (i + 1)]

            batch_inputs = input_tensor[batch_index]
            batch_targets = targets[batch_index].unsqueeze(1)

            values = model(batch_inputs)
            loss = criterion(values, batch_targets)
            total_loss += abs(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return total_loss
