import torch
import torch.nn as nn
from torch import optim
import gym
import numpy as np


def init_weights(m):
    """
    Initiliazes the weights of the model by using kaiming normal initialization.
    :param m: Handle for the model
    :return:
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


class DQNModel(torch.nn.Module):

    def __init__(self, env: gym.Env, discrete_actions: np.ndarray, lr: float = 1e-3, optimizer: str = 'adam'):
        """
        Constructor
        :param env: Gym environment object
        :param discrete_actions: Numpy array which stores the values for the action bins
        :param lr: Learning rate to use
        :param optimizer: Optimizer to use
        """

        super(DQNModel, self).__init__()

        self.discrete_actions = discrete_actions
        self.n_outputs = len(discrete_actions)
        self.n_inputs = env.observation_space.shape[0]

        self.model = self.get_model()

        # initialize the weights
        self.apply(init_weights)
        self.train()

        if optimizer == 'adam': # adam used
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'rmsprop':
            self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = None

    def get_model(self):
        """
        Returns a model which specifies the architecture of the network
        """
        return nn.Sequential(
            nn.Linear(self.n_inputs, self.n_outputs)
        )

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

        return out.float()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Runs forward pass in the current model given input data X
        :param X: Input data as numpy array
        :return: Output value of the model as numpy as array
        """
        self.eval()
        X = torch.from_numpy(X)
        with torch.no_grad():
            return self(X).detach().numpy()
