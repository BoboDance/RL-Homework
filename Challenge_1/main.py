import numpy as np
import sklearn

from Challenge_1.DataGenerator import DataGenerator
from Challenge_1.ModelsGP import GPModel
import logging

from Challenge_1.util.ColorLogger import enable_color_logging

enable_color_logging(debug_lvl=logging.DEBUG)

seed = 123
n_samples_train = 100
n_samples_test = 100

env_name = "Pendulum-v0"
#env_name = "Qube-v0"

dg_train = DataGenerator(env_name=env_name, seed=seed)
# s - state
# a - action
# r - reward
# s_prime - future state after you taken the action from state s
s_prime, s, a, r = dg_train.get_samples(n_samples_train)

# create training input pairs
s_a_pairs = np.concatenate([s, a[:, np.newaxis]], axis=1)

# solve regression problem s_prime = f(s,a)
dynamics_model = GPModel()
dynamics_model.fit(s_a_pairs, s_prime)

# solve regression problem r = g(s,a)
reward_model = GPModel()
reward_model.fit(s_a_pairs, r)

# --------------------------------------------------------------
# compute accuracy on test set

dg_test = DataGenerator(env_name=env_name, seed=seed)
s_prime_test, s_test, a_test, r_test = dg_test.get_samples(n_samples_test)

# create test input pairs
s_a_pairs_test = np.concatenate([s_test, a_test[:, np.newaxis]], axis=1)

# make prediction for dynamics model
s_a_pred = dynamics_model.predict(s_a_pairs_test)
# make prediction for reward model
reward_pred = reward_model.predict(s_a_pairs_test)

mse_dynamics = ((s_prime_test - s_a_pred) ** 2).mean(axis=0)
logging.debug('rtest: %s - reward_pred: %s' % (s_prime_test.shape, s_prime_test.shape))
mse_reward = ((r_test - reward_pred) ** 2).mean() # same as sklearn.metrics.mean_squared_error()
print("MSE for dynamics model: {}".format(mse_dynamics))
print("MSE for reward model: {}".format(mse_reward))
