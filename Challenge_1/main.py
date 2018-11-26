import logging

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.envs.classic_control import PendulumEnv

from Challenge_1.DataGenerator import DataGenerator
from Challenge_1.Discretizer import Discretizer
from Challenge_1.ModelsGP import GPModel
from Challenge_1.PolicyIteration import PolicyIteration
from Challenge_1.ValueIteration import ValueIteration
import Challenge_1.custom_env
from Challenge_1.util.ColorLogger import enable_color_logging

enable_color_logging(debug_lvl=logging.DEBUG)

seed = 1234

# env_name = "Pendulum-v0"
env_name = "PendulumCustom-v0"
# env_name = "Qube-v0"
# env_name = "MountainCarContinuous-v0"

def start_policy_iteration(env_name, algorithm, n_samples=400, bins_state=10, bins_action=20, seed=1, theta=1e-3):

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

    discretizer_state = Discretizer(n_bins=bins_state, space=env.observation_space)
    discretizer_action = Discretizer(n_bins=bins_action, space=env.action_space)

    if algorithm == "pi":
        algo = PolicyIteration(env=env, dynamics_model=dynamics_model, reward_model=reward_model,
                               discretizer_state=discretizer_state, discretizer_action=discretizer_action, theta=theta)
    elif algorithm == "vi":
        algo = ValueIteration(env=env, dynamics_model=dynamics_model, reward_model=reward_model,
                              discretizer_state=discretizer_state, discretizer_action=discretizer_action, theta=theta)
    else:
        raise NotImplementedError()

    algo.run()

    return algo.policy, discretizer_state


def test_run(env_name, policy, discretizer_state, n_episodes=100):
    print(policy)
    env = gym.make(env_name)
    rewards = np.zeros(n_episodes)

    for i in range(n_episodes):
        done = False
        state = env.reset()

        while not done:
            env.render()
            state = discretizer_state.discretize(np.atleast_2d(state))
            action = policy[tuple(state.T)]  # TODO: REMOVE THIS HARDCODING AND MAKE IT MORE GENRAL
            action = action - (20 - 1) / 2
            action = action / ((20 - 1) / (2 * 2))
            state, reward, done, _ = env.step(action)
            rewards[i] += reward

        print("Intermediate reward: {}".format(rewards[i]))

    print("Mean reward over {} epochs: {}".format(n_episodes, rewards.mean()))


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
policy, discretizer_state = start_policy_iteration(env_name, algorithm="pi", seed=seed)
test_run(env_name, policy, discretizer_state)
