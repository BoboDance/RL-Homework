import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ContinuousPolicy(nn.Module):
    def __init__(self, env, n_hidden_units=16):
        """
        Create a continuous policy which samples its actions using predicted a gaussian for a given observation.

        :param env: Gym environment object for which this policy object is created
        :param n_hidden_units: Number of nodes in the hidden layer
        """

        super(ContinuousPolicy, self).__init__()

        self.n_inputs = env.observation_space.shape[0]
        self.n_actions = env.action_space.shape[0]
        self.action_space = env.action_space

        self.linear_input_to_hidden = nn.Linear(self.n_inputs, n_hidden_units)
        self.linear_hidden_to_mean = nn.Linear(n_hidden_units, self.n_actions)
        # self.linear_hidden_to_sigma = nn.Linear(n_hidden_units, self.n_actions)
        self.sigma = nn.Parameter(torch.zeros(self.n_actions))

        self.train()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: means and sigma_squared for each action
        """
        hidden = F.relu(self.linear_input_to_hidden(inputs))
        means = self.linear_hidden_to_mean(hidden)
        # sigma = self.linear_hidden_to_sigma(hidden)
        # sigma = F.softplus(sigma)

        return means, F.softplus(self.sigma)

    def choose_action_by_sampling(self, observation):
        """
        Sample a random action according to our current policy.

        :param observation: the current observation
        :return: the chosen action and the corresponding confidence
        """
        # convert the given observation into a torch tensor
        # print(observation)
        observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        # get the probability distribution from the neural net
        means, sigma = self.forward(observation)
        # define a normal distribution and sample from it
        distribution = Normal(means.cpu(), sigma.cpu())
        action = distribution.sample()
        # confidence is prod(p1, ..., pn), so we get log(prod(p1, ..., pn)) = sum(log(p1, ..., pn))
        log_confidence = distribution.log_prob(action).sum()

        return action.detach().numpy().flatten(), log_confidence