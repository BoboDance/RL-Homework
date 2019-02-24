import torch
import torch.nn as nn
from torch.distributions import Categorical

from Challenge_3.Util import init_weights


class DiscretePolicy(nn.Module):
    def __init__(self, env,  discrete_actions, n_hidden_units = 16):
        """
        Create a discrete policy which samples its actions from a categorical distribution corresponding to the given
        discrete actions.

        :param env: Gym environment object
        :param discrete_actions: Number of discrete actions (will specify output of the network)
        :param n_hidden_units: Number of nodes in the hidden layer
        """

        super(DiscretePolicy, self).__init__()

        self.n_inputs = env.observation_space.shape[0]
        self.discrete_actions = discrete_actions
        self.n_outputs = self.discrete_actions.shape[0]
        self.n_hidden_units = n_hidden_units

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_hidden_units),
            nn.ReLU(),
            nn.Linear(self.n_hidden_units, self.n_outputs),
            nn.Softmax(1)
        )

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
        Sample a random action according to our current policy.

        :param observation: the current observation
        :return: the chosen action and the corresponding confidence
        """
        # convert the given observation into a torch tensor
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        # get the probability distribution from the neural net
        probabilities = self.forward(observation).cpu()
        # define a categorical distribution and sample from it
        distribution = Categorical(probabilities)
        sampled_action = distribution.sample()

        log_prob = distribution.log_prob(sampled_action)
        action = self.discrete_actions[sampled_action.detach().numpy()]
        return action, log_prob
