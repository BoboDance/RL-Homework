import itertools
import logging

import gym
import matplotlib.pyplot as plt
import numpy as np
import quanser_robots

from Challenge_1.Algorithms.PolicyIteration import PolicyIteration
from Challenge_1.Algorithms.ValueIteration import ValueIteration
from Challenge_1.EnvironmentModels.NNModel import NNModel
from Challenge_1.EnvironmentModels.SklearnModel import SklearnModel
from Challenge_1.util.ColorLogger import enable_color_logging
from Challenge_1.util.DataGenerator import DataGenerator
from Challenge_1.util.Discretizer import Discretizer

enable_color_logging(debug_lvl=logging.INFO)

seed = 1234
# avoid auto removal of import with pycharm
quanser_robots

env_name = "Pendulum-v2"
# env_name = "PendulumCustom-v0"
# env_name = "MountainCarContinuous-v0"
# env_name = "Qube-v0"


def grid_search(env_name, seed, dim=2, algo="pi"):
    for dense_loc in list(itertools.product(["center", "edge", "start", "end"], repeat=dim)) + [None]:
        for MC_samples in [1, 10, 25, 50, 100, 250, 500, 1000]:
            for state_bins in [2, 10, 26, 50, 76, 100, 150]:
                policy, discretizer_action, discretizer_state = start_policy_iteration(env_name, algorithm=algo,
                                                                                       n_samples=10000,
                                                                                       bins_state=state_bins,
                                                                                       bins_action=2,
                                                                                       seed=seed, theta=1e-9,
                                                                                       use_MC=True,
                                                                                       MC_samples=MC_samples,
                                                                                       dense_location=dense_loc)
                print("Score dense_loc: {} -- MC samples: {} -- state bins: {} ".format(dense_loc, MC_samples,
                                                                                        state_bins))
                test_run(env_name, policy, discretizer_action, discretizer_state)


# best for pendulum: 500 MC samples, 50 bins, [center, edge]
# TODO: only use equal bins numbers
# ["center", "center", "center", "center"]
def start_policy_iteration(env_name, algorithm="pi", n_samples=10000, bins_state=150, bins_action=2, seed=1,
                           theta=1e-3, use_MC=True, MC_samples=1, dense_location=["center", "edge"]):
    env = gym.make(env_name)
    print("Training with {} samples.".format(n_samples))

    dg_train = DataGenerator(env_name=env_name, seed=seed)

    # s_prime - future state after you taken the action from state s
    state_prime, state, action, reward = dg_train.get_samples(n_samples)

    # create training input pairs
    s_a_pairs = np.concatenate([state, action[:, np.newaxis]], axis=1)

    # # solve regression problem s_prime = f(s,a)
    # dynamics_model = SklearnModel(type="rf")
    # dynamics_model.fit(s_a_pairs, state_prime)
    #
    # # solve regression problem r = g(s,a)
    # reward_model = SklearnModel(type="rf")
    # reward_model.fit(s_a_pairs, reward)

    # But performance should not change much
    dynamics_model = NNModel(n_inputs=env.observation_space.shape[0] + env.action_space.shape[0],
                             n_outputs=env.observation_space.shape[0],
                             scaling=env.observation_space.high)

    reward_model = NNModel(n_inputs=env.observation_space.shape[0] + env.action_space.shape[0],
                           n_outputs=1,
                           scaling=None)

    dynamics_model.load_model("./NN-state_dict_dynamics_10000_200hidden")
    reward_model.load_model("./NN-state_dict_reward_10000_200hidden")

    # center, edge for pendulum is best for RF
    # edge, center is best for NN
    discretizer_state = Discretizer(n_bins=bins_state, space=env.observation_space,
                                    dense_locations=dense_location)
    discretizer_action = Discretizer(n_bins=bins_action, space=env.action_space)

    if algorithm == "pi":
        algo = PolicyIteration(env=env, dynamics_model=dynamics_model, reward_model=reward_model,
                               discretizer_state=discretizer_state, discretizer_action=discretizer_action, theta=theta,
                               use_MC=use_MC, MC_samples=MC_samples)
    elif algorithm == "vi":
        algo = ValueIteration(env=env, dynamics_model=dynamics_model, reward_model=reward_model,
                              discretizer_state=discretizer_state, discretizer_action=discretizer_action, theta=theta,
                              use_MC=use_MC, MC_samples=MC_samples)
    else:
        raise NotImplementedError()

    algo.run(max_iter=100)

    return algo.policy, discretizer_action, discretizer_state


def test_run(env_name, policy, discretizer_action, discretizer_state, n_episodes=100):
    env = gym.make(env_name)

    # if len(policy.shape) == 2:
    #     plt.matshow(policy)
    #     plt.colorbar()
    #     plt.title("Policy for {}".format(env_name))
    #     plt.show()

    rewards = np.zeros(n_episodes)

    for i in range(n_episodes):
        done = False
        env.seed(i)
        state = env.reset()

        while not done:
            # env.render()
            state = discretizer_state.discretize(np.atleast_2d(state))
            action = policy[tuple(state.T)]
            state, reward, done, _ = env.step(action)
            rewards[i] += reward

        # print("Intermediate reward: {}".format(rewards[i]))

    print("Mean reward over {} epochs: {}".format(n_episodes, rewards.mean()))


def train_and_eval_nn(train=True, n_samples=25000, n_steps=20000):
    env = gym.make(env_name)
    path = "./NN-state_dict"

    dynamics_model = NNModel(n_inputs=env.observation_space.shape[0] + env.action_space.shape[0],
                             n_outputs=env.observation_space.shape[0],
                             scaling=env.observation_space.high)

    reward_model = NNModel(n_inputs=env.observation_space.shape[0] + env.action_space.shape[0],
                           n_outputs=1,
                           scaling=None)

    if train:

        dg_train = DataGenerator(env_name=env_name, seed=seed)

        # s_prime - future state after you taken the action from state s
        state_prime, state, action, reward = dg_train.get_samples(n_samples)

        # create training input pairs
        s_a_pairs = np.concatenate([state, action[:, np.newaxis]], axis=1).reshape(-1, env.observation_space.shape[0] +
                                                                                   env.action_space.shape[0])
        reward = reward.reshape(-1, 1)
        state_prime = state_prime.reshape(-1, env.observation_space.shape[0])

        dynamics_model.train_network(s_a_pairs, state_prime, n_steps, path + "_dynamics")
        reward_model.train_network(s_a_pairs, reward, n_steps, path + "_reward")
    else:
        dynamics_model.load_model(path + "_dynamics")
        reward_model.load_model(path + "_reward")

    dg_test = DataGenerator(env_name=env_name, seed=seed + 1)
    s_prime, s, a, r = dg_test.get_samples(n_samples)

    # create test input pairs
    s_a = np.concatenate([s, a[:, np.newaxis]], axis=1).reshape(-1, env.observation_space.shape[0] +
                                                                env.action_space.shape[0])
    r = r.reshape(-1, 1)
    s_prime = s_prime.reshape(-1, env.observation_space.shape[0])

    dynamics_model.validate_model(s_a, s_prime)
    reward_model.validate_model(s_a, r)


def find_good_sample_size(env_name, seed, steps=1000, max=25000, n_samples_test=25000):
    dyn_history_test = []
    rwd_history_test = []
    rwd_history_train = []
    dyn_history_train = []

    data_point_range = range(steps, max + 1, steps)

    dg_test = DataGenerator(env_name=env_name, seed=seed + 1)
    s_prime_test, s_test, a_test, r_test = dg_test.get_samples(n_samples_test)

    # create test input pairs
    s_a_pairs_test = np.concatenate([s_test, a_test[:, np.newaxis]], axis=1)

    for i in data_point_range:
        print("Training with {} samples.".format(i))
        n_samples_train = i

        dg_train = DataGenerator(env_name=env_name, seed=seed)

        # s_prime - future state after you taken the action from state s
        state_prime, state, action, reward = dg_train.get_samples(n_samples_train)

        # create training input pairs
        s_a_pairs = np.concatenate([state, action[:, np.newaxis]], axis=1)

        # solve regression problem s_prime = f(s,a)
        dynamics_model = SklearnModel()
        dynamics_model.fit(s_a_pairs, state_prime)

        # solve regression problem r = g(s,a)
        reward_model = SklearnModel()
        reward_model.fit(s_a_pairs, reward)

        # --------------------------------------------------------------
        # compute accuracy on test set

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


grid_search(env_name, seed, 2, "pi")
# find_good_sample_size(env_name, seed)
# train_and_eval_nn(train=True)
# policy, discretizer_action, discretizer_state = start_policy_iteration(env_name, seed=seed)
# test_run(env_name, policy, discretizer_action, discretizer_state)
#
