import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0)


act = nn.ReLU()


class NNModelPendulum(torch.nn.Module):

    def __init__(self, n_inputs, n_outputs, scaling=None, lr=1e-3, optimizer='adam'):
        super(NNModelPendulum, self).__init__()

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
        hidden = 200

        self.model = nn.Sequential(
            nn.Linear(self.n_inputs, hidden),
            act,
            nn.Linear(hidden, hidden),
            act,
            nn.Linear(hidden, hidden),
            act,
            nn.Linear(hidden, hidden),
            act,
            nn.Linear(hidden, self.n_outputs),
        )

        # initialize the weights
        self.apply(init_weights)
        self.train()

        if optimizer == 'adam':
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        elif optimizer == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def forward(self, inputs):
        """
        Defines the forward pass of the network.

        :param inputs: Input array object which sufficiently represents the full state of the environment.
        :return: reward, mu, sigma
        """
        inputs = inputs.float()

        """"
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        """

        out = self.model(inputs)

        if self.scaling is not None:
            out = torch.from_numpy(self.scaling) * torch.tanh(out)  # self.out(x))
        # else:
        #    out = self.out(x)

        return out

    def train_network(self, X, y, steps, path):

        y = torch.from_numpy(y).float()
        X = torch.from_numpy(X).float()

        criterion = nn.MSELoss()

        history = {}
        history['loss'] = []

        for i in range(steps):
            self.optimizer.zero_grad()

            out = self(X)

            loss = criterion(out, y)

            print("Step: {:d} -- total loss: {:3.8f}".format(i, loss.item()))

            history['loss'].append(loss.item())
            loss.backward()

            self.optimizer.step()

        torch.save(self.state_dict(), path)
        torch.save(self, path + "_model")

        return history

    def train_net(self, X, y, path, n_epochs=50, batch_size=32):
        # https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way

        train_loss = []
        # val_loss = []

        y = torch.from_numpy(y).float()
        X = torch.from_numpy(X).float()

        lossfunction = nn.MSELoss()

        for epoch in range(n_epochs):

            # X is a torch Variable
            permutation = torch.randperm(X.size()[0])

            for i in range(0, X.size()[0], batch_size):
                self.optimizer.zero_grad()

                indices = permutation[i:i + batch_size]
                batch_x, batch_y = X[indices], y[indices]

                # in case you wanted a semi-full example
                outputs = self(batch_x)
                loss = lossfunction(outputs, batch_y)

                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            # if epoch % 25 == 0:
            #    for g in optimizer.param_groups:
            #        g['lr'] /= 10

            print("Step: {:d} -- total loss: {:3.8f}".format(epoch, loss.item()))
            # val_loss.append(self.validate_model(X_val, Y_val))

        torch.save(self.state_dict(), path)

    def validate_model(self, X, y):

        self.eval()

        with torch.no_grad():
            # y = torch.from_numpy(y).float()
            X = torch.from_numpy(X).float()

            out = self(X)

            mse_test = ((out.detach().numpy() - y) ** 2).mean(axis=0)

            print("Test MSE: {}".format(mse_test.mean()))

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()

    def predict(self, X):
        self.eval()
        X = torch.from_numpy(X)
        return self(X).detach().numpy()
