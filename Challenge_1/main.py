import logging

import gym
import quanser_robots
import matplotlib.pyplot as plt
import numpy as np

from Challenge_1.DataGenerator import DataGenerator
from Challenge_1.Discretizer import Discretizer
from Challenge_1.ModelsGP import GPModel
from Challenge_1.PolicyIteration import PolicyIteration
from Challenge_1.util.ColorLogger import enable_color_logging

enable_color_logging(debug_lvl=logging.DEBUG)

seed = 123

env_name = "Pendulum-v0"
env_name = "Qube-v0"


def start_policy_iteration(env_name, n_samples=400, bins=50, seed=1):
    env = gym.make(env_name)
    print("Training with {} samples.".format(n_samples))

    dg_train = DataGenerator(env_name=env_name, seed=seed)

    # s_prime - future state after you taken the action from state s
    state_prime, state, action, reward = dg_train.get_samples(n_samples)

    # create training input pairs
    s_a_pairs = np.concatenate([state, action[:, np.newaxis]], axis=1)

    # solve regression problem s_prime = f(s,a)
    dynamics_model = GPModel()
    dynamics_model.fit(s_a_pairs, state_prime)

    # solve regression problem r = g(s,a)
    reward_model = GPModel()
    reward_model.fit(s_a_pairs, reward)

    discretizer_state = Discretizer(n_bins=bins, space=env.observation_space)
    discretizer_action = Discretizer(n_bins=bins, space=env.action_space)

    pi = PolicyIteration(env=env, dynamics_model=dynamics_model, reward_model=reward_model,
                         discretizer_state=discretizer_state, discretizer_action=discretizer_action)

    pi.run()

    return pi


def test_run(env_name, policy):
    env = gym.make(env_name)
    state = env.reset()
    done = False

    r = 0
    t = 0

    while not done:
        t += 1
        action = policy[state[0]][state[1]][state[2]]
        state, reward, done, _ = env.step(action)
        r += reward

    print(t, r)


def find_good_sample_size(env_name, seed):
    dyn_history_test = []
    rwd_history_test = []
    rwd_history_train = []
    dyn_history_train = []

    data_point_range = range(100, 2001, 100)

    for i in data_point_range:
        print("Training with {} samples.".format(i))
        n_samples_train = i
        n_samples_test = i

        dg_train = DataGenerator(env_name=env_name, seed=seed)

        # s_prime - future state after you taken the action from state s
        state_prime, state, action, reward = dg_train.get_samples(n_samples_train)

        # create training input pairs
        s_a_pairs = np.concatenate([state, action[:, np.newaxis]], axis=1)

        # solve regression problem s_prime = f(s,a)
        dynamics_model = GPModel()
        dynamics_model.fit(s_a_pairs, state_prime)

        # solve regression problem r = g(s,a)
        reward_model = GPModel()
        reward_model.fit(s_a_pairs, reward)

        # --------------------------------------------------------------
        # compute accuracy on test set

        dg_test = DataGenerator(env_name=env_name, seed=seed)
        s_prime_test, s_test, a_test, r_test = dg_test.get_samples(n_samples_test)

        # create test input pairs
        s_a_pairs_test = np.concatenate([s_test, a_test[:, np.newaxis]], axis=1)

        # make prediction for dynamics model
        s_a_pred_train = dynamics_model.predict(s_a_pairs)
        s_a_pred_test = dynamics_model.predict(s_a_pairs_test)

        # make prediction for reward model
        reward_pred_train = reward_model.predict(s_a_pairs)
        reward_pred_test = reward_model.predict(s_a_pairs_test)

        # logging.debug('rtest: %s - reward_pred: %s' % (s_prime_test.shape, s_prime_test.shape))

        # --------------------------------------------------------------

        # same as sklearn.metrics.mean_squared_error()

        # train
        mse_dynamics_train = ((state_prime - s_a_pred_train) ** 2).mean(axis=0)
        mse_reward_train = ((reward - reward_pred_train) ** 2).mean()

        # test
        mse_dynamics_test = ((s_prime_test - s_a_pred_test) ** 2).mean(axis=0)
        mse_reward_test = ((r_test - reward_pred_test) ** 2).mean()

        print("Test MSE for dynamics model: {}".format(mse_dynamics_test))
        print("Test MSE for reward model: {}".format(mse_reward_test))

        dyn_history_test.append(mse_dynamics_test)
        rwd_history_test.append(mse_reward_test)
        dyn_history_train.append(mse_dynamics_train)
        rwd_history_train.append(mse_reward_train)

    dyn_history_test = np.array(dyn_history_test)
    dyn_history_train = np.array(dyn_history_train)
    rwd_history_test = np.array(rwd_history_test)
    rwd_history_train = np.array(rwd_history_train)

    for i in range(dyn_history_test.shape[1]):
        plt.figure(i)
        plt.plot(data_point_range, dyn_history_test[:, i], label="Test")
        plt.plot(data_point_range, dyn_history_train[:, i], label="Train")
        plt.xlabel("# Samples")
        plt.ylabel("MSE")
        plt.title("Dynamics Performance (State_{}) for different Sample Sizes".format(i))
        plt.legend()

    plt.figure(dyn_history_test.shape[1] + 1)
    plt.plot(data_point_range, rwd_history_test, label="Test")
    plt.plot(data_point_range, rwd_history_train, label="Train")
    plt.xlabel("# Samples")
    plt.ylabel("MSE")
    plt.title("Reward Performance for different Sample Sizes")
    plt.legend()

    plt.show()


# find_good_sample_size(env_name, seed)
pi = start_policy_iteration(env_name, seed=seed, bins=10)
test_run(env_name, pi)
