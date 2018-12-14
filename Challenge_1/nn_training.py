
# coding: utf-8

# ## Neural Network Training
# 
# * In this Jupyter Notebook both the dynamic models and reward models are trained and
# later exported as a paramter dictionary for later usage in with pytorch
# * We use a neural network for both dynamics and reward
# * You can define one of the two environemnts in the `Settings` block

# In[1]:


import logging

import gym
import matplotlib.pyplot as plt
import numpy as np
import quanser_robots

import sys
sys.path.insert(0,'../')
from Challenge_1.Algorithms.PolicyIteration import PolicyIteration
from Challenge_1.Algorithms.ValueIteration import ValueIteration
from Challenge_1.Models.NNModelPendulum import NNModelPendulum
from Challenge_1.Models.NNModelQube import NNModelQube
from Challenge_1.Models.SklearnModel import SklearnModel
from Challenge_1.util.ColorLogger import enable_color_logging
from Challenge_1.util.DataGenerator import DataGenerator
from Challenge_1.util.Discretizer import Discretizer
from Challenge_1.util.state_preprocessing import reconvert_state_to_angle, normalize_input, get_feature_space_boundaries, convert_state_to_sin_cos
import itertools
from torch.optim.lr_scheduler import *
enable_color_logging(debug_lvl=logging.INFO)
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
import torch.nn as nn
import torch
import torch.optim as optim
import logging


def create_dataset(env, env_name, seed, n_samples, angle_features, convert_to_sincos=False):
    """
    Creates the dataset for training the NN
    """
    
    dg_train = DataGenerator(env_name=env_name, seed=seed)

    # s_prime - future state after you taken the action from state s
    state_prime, state, action, reward = dg_train.get_samples(n_samples)

    if convert_to_sincos is True:
        state = convert_state_to_sin_cos(state, angle_features)
        state_prime = convert_state_to_sin_cos(state_prime, angle_features)
    
    # create training input pairs
    s_a_pairs = np.concatenate([state, action[:, np.newaxis]], axis=1).reshape(-1, state.shape[1] +
                                                                               env.action_space.shape[0])
    reward = reward.reshape(-1, 1)

    return s_a_pairs, state_prime, reward


def validate_model(model, X, y):

    model.eval()

    with torch.no_grad():

        out = model(X)

        mse_test = ((out.detach().numpy() - y) ** 2).mean(axis=0)

        logging.debug("Test MSE: {}".format(mse_test))
        logging.debug("Test MSE (mean): {}".format(mse_test.mean()))

    return mse_test.mean()


# In[ ]:


def train(model, optimizer, X, Y, X_val, Y_val, n_epochs=150, batch_size=32, lossfunction=nn.MSELoss()):
    
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()

    X_val = torch.from_numpy(X_val)
    Y_val = Y_val

    # https://stackoverflow.com/questions/45113245/how-to-get-mini-batches-in-pytorch-in-a-clean-and-efficient-way

    train_loss = []
    val_loss = []
    for epoch in range(n_epochs):

        # X is a torch Variable
        permutation = torch.randperm(X.size()[0])

        for i in range(0,X.size()[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X[indices], Y[indices]

            # in case you wanted a semi-full example
            outputs = model.forward(batch_x)
            loss = lossfunction(outputs, batch_y)

            loss.backward()
            optimizer.step()

        if epoch % 50 == 0:
            for g in optimizer.param_groups:
                g['lr'] /= 2

        logging.debug("Epoch: {:d} -- total loss: {:3.8f}".format(epoch+1, loss.item()))
        train_loss.append(loss.item())
        val_loss.append(validate_model(model, X_val, Y_val))

    return train_loss, val_loss


# ## Start the training process

# ## Train the Dynamics Model

# In[ ]:

