import itertools
import logging

import gym
import matplotlib.pyplot as plt
import numpy as np
import quanser_robots
import scipy

from Challenge_1.Algorithms.PolicyIteration import PolicyIteration
from Challenge_1.Algorithms.ValueIteration import ValueIteration
from Challenge_1.Models.NNModel import NNModel
from Challenge_1.Models.NNModelQube import NNModelQube
from Challenge_1.Models.SklearnModel import SklearnModel
from Challenge_1.util.ColorLogger import enable_color_logging
from Challenge_1.util.DataGenerator import DataGenerator
from Challenge_1.util.Discretizer import Discretizer
from Challenge_1.util.state_preprocessing import get_feature_space_boundaries

enable_color_logging(debug_lvl=logging.INFO)

seed = 1234
# avoid auto removal of import with pycharm
quanser_robots

env_name = "Pendulum-v2"
#env_name = "Qube-v0"

# index list of angle features
if env_name == 'Pendulum-v2':
    angle_features = [0]  # Pendulum-v2
    dynamics_model_params = "./Weights/model_dynamics_Pendulum-v2_mse_0.00002354.params"
    reward_model_params = "./Weights/model_reward_Pendulum-v2_mse_0.00581975.params"
elif env_name == "Qube-v0":
    angle_features = [0, 1]  # Qube-v0
    dynamics_model_params = "./Weights/model_dynamics_Qube-v0_mse_0.00026449.params"
    reward_model_params = "./Weights/model_reward_Qube-v0_mse_0.00001899.params"


def main():
    random_search(env_name, seed, 4)
    # grid_search(env_name, seed, 4, "vi")
    # find_good_sample_size(env_name, seed, steps=100, max=2000, n_samples_test=2000, type='gp')
    # train_and_eval_nn(train=False)
    # best for pendulum VI: 500 MC samples, 50 bins, [center, edge] -- reward: 334
    # best for pendulum PI: ('edge', 'center') -- MC samples: 500 -- state bins: 100 --- reward: 330
    # bins_state = [100] * 4
    #
    # policy, discretizer_action, discretizer_state = run(env_name, algorithm="vi",
    #                                                     n_samples=10000,
    #                                                     bins_state=[],
    #                                                     bins_action=[2],
    #                                                     seed=seed, theta=1e-9,
    #                                                     use_MC=True,
    #                                                     MC_samples=1,
    #                                                     dense_location=None,
    #                                                     dynamics_model_params=dynamics_model_params,
    #                                                     reward_model_params=reward_model_params,
    #                                                     angle_features=angle_features)
    # test_run(env_name, policy, discretizer_action, discretizer_state, n_episodes=1000)


def random_search(env_name, seed, dim=2, algo="vi", n_steps=1000):
    choices = [None] + list(itertools.product(["center", "edge", "start", "end"], repeat=dim))

    for i in range(n_steps):
        bins = np.random.randint(5, 81, size=dim)
        bins = bins + bins % 2
        dense_loc = np.random.choice(choices)
        MC_samples = 1
        policy, discretizer_action, discretizer_state = run(env_name, seed=seed, bins_state=bins,
                                                             bins_action=[2], angle_features=angle_features,
                                                             MC_samples=1, dense_location=None,
                                                             dynamics_model_params=dynamics_model_params,
                                                             reward_model_params=reward_model_params)

        string = "Run: {} -- Score dense_loc: {} -- MC samples: {} -- state bins: {}".format(i, dense_loc,
                                                                                                MC_samples, bins)
        print(string)

        # policy, discretizer_action, discretizer_state = run(env_name, algorithm=algo,
        #                                                     n_samples=25000,
        #                                                     bins_state=bins,
        #                                                     bins_action=[2],
        #                                                     seed=seed, theta=1e-9,
        #                                                     use_MC=True,
        #                                                     MC_samples=MC_samples,
        #                                                     dense_location=dense_loc,
        #                                                     dynamics_model_params=dynamics_model_params,
        #                                                     reward_model_params=reward_model_params,
        #                                                     angle_features=angle_features)

        f = open("Results_qube.txt", "a+")
        f.write(string)
        f.close()

        test_run(env_name, policy, discretizer_action, discretizer_state)


def grid_search(env_name, seed, dim=2, algo="pi"):
    for dense_loc in [None] + list(itertools.product(["center", "edge", "start", "end"], repeat=dim)):
        for MC_samples in [1, 10, 25, 50, 100, 500]:
            for state_bins in [2, 10, 26, 50, 76, 100, 150, 200]:
                policy, discretizer_action, discretizer_state = run(env_name, algorithm=algo,
                                                                    n_samples=10000,
                                                                    bins_state=[state_bins] * dim,
                                                                    bins_action=[2],
                                                                    seed=seed, theta=1e-9,
                                                                    use_MC=True,
                                                                    MC_samples=MC_samples,
                                                                    dense_location=dense_loc,
                                                                    dynamics_model_params=dynamics_model_params,
                                                                    reward_model_params=reward_model_params,
                                                                    angle_features=angle_features)
                print("Score dense_loc: {} -- MC samples: {} -- state bins: {} ".format(dense_loc, MC_samples,
                                                                                        state_bins))
                test_run(env_name, policy, discretizer_action, discretizer_state)


# best for pendulum VI: 500 MC samples, 50 bins, [center, edge] -- reward: 334
# best for pendulum PI: ('edge', 'center') -- MC samples: 500 -- state bins: 100 --- reward: 330
# TODO: only use equal bins numbers
# ["center", "center", "center", "center"]
def run(env_name, dense_location, angle_features, dynamics_model_params, reward_model_params, algorithm="pi",
        n_samples=10000, bins_state=[100, 100], bins_action=[3], seed=1, theta=1e-9, use_MC=True, MC_samples=1, ):
    env = gym.make(env_name)
    print("Training with {} samples.".format(n_samples))

    dg_train = DataGenerator(env_name=env_name)

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
    n_inputs = env.observation_space.shape[0] + env.action_space.shape[0] + len(angle_features)
    n_outputs = env.observation_space.shape[0] + len(angle_features)

    # create both boundaries of the feature space
    x_low, x_high = get_feature_space_boundaries(env, angle_features)
    # scaling defines how our outputs will be scaled after the tanh function
    # for this we use all state features ergo all of X_high excluding the last action feature
    scaling = x_high[:-1]

    if env_name == 'Pendulum-v2':
        dynamics_model = NNModel(n_inputs=n_inputs,
                                 n_outputs=n_outputs,
                                 scaling=scaling)

        reward_model = NNModel(n_inputs=n_inputs,
                               n_outputs=1,
                               scaling=None)
    elif env_name == 'Qube-v0':
        dynamics_model = NNModelQube(n_inputs=n_inputs,
                                     n_outputs=n_outputs,
                                     scaling=scaling)

        reward_model = NNModelQube(n_inputs=n_inputs,
                                   n_outputs=1,
                                   scaling=None)
    # print('dynamics mdoel')
    # print(dynamics_model)

    dynamics_model.load_model(dynamics_model_params)
    reward_model.load_model(reward_model_params)

    # dynamics_model.load_model("./NN-state_dict_dynamics_10000_large")
    # reward_model.load_model("./NN-state_dict_reward_10000_large")

    # center, edge for pendulum is best for RF
    # edge, center is best for NN
    discretizer_state = Discretizer(n_bins_per_feature=bins_state, space=env.observation_space,
                                    dense_locations=dense_location)
    discretizer_action = Discretizer(n_bins_per_feature=bins_action, space=env.action_space)

    if algorithm == "pi":
        algo = PolicyIteration(env=env, dynamics_model=dynamics_model, reward_model=reward_model,
                               discretizer_state=discretizer_state, discretizer_action=discretizer_action, theta=theta,
                               use_MC=use_MC, MC_samples=MC_samples, angle_features=angle_features, verbose=False)
    elif algorithm == "vi":
        algo = ValueIteration(env=env, dynamics_model=dynamics_model, reward_model=reward_model,
                              discretizer_state=discretizer_state, discretizer_action=discretizer_action, theta=theta,
                              use_MC=use_MC, MC_samples=MC_samples, angle_features=angle_features, verbose=False)
    else:
        raise NotImplementedError()

    algo.run(max_iter=100)

    return algo.policy, discretizer_action, discretizer_state


def test_run(env_name, policy, discretizer_action, discretizer_state, n_episodes=1000):
    env = gym.make(env_name)

    # policy = scipy.ndimage.filters.gaussian_filter(policy, 3)
    policy = scipy.signal.medfilt(policy)

    if len(policy.shape) == 2:
        plt.imshow(policy)
        plt.colorbar()
        plt.title("Policy for {}".format(env_name))
        plt.show()

    rewards = np.zeros(n_episodes)

    for i in range(n_episodes):
        done = False
        env.seed(i*seed)
        state = env.reset()

        while not done:
            # env.render()
            state = discretizer_state.discretize(np.atleast_2d(state))
            action = policy[tuple(state.T)]
            state, reward, done, _ = env.step(action)
            rewards[i] += reward
        # print("Intermediate reward: {}".format(rewards[i]))

    print("Mean reward over {} epochs: {}".format(n_episodes, rewards.mean()))
    print("Mean reward over first 100 epochs: {}".format(rewards[:100].mean()))


def find_good_sample_size(env_name, seed, steps=1000, max=25000, n_samples_test=25000, type='rf'):
    dyn_history_test = []
    rwd_history_test = []
    rwd_history_train = []
    dyn_history_train = []

    data_point_range = range(steps, max + 1, steps)

    dg_test = DataGenerator(env_name=env_name, seed=seed + 1)
    s_prime_test, s_test, a_test, r_test = dg_test.get_samples(n_samples_test)

    # create test input pairs
    s_a_pairs_test = np.concatenate([s_test, a_test[:, np.newaxis]], axis=1)

    if type == 'rf':
        model_name = 'Random Forest'
    elif type == 'gp':
        model_name = 'Gaussian Process'
    else:
        raise Exception('Unsupported model type given')

    for i in data_point_range:
        print("Training with {} samples.".format(i))
        n_samples_train = i

        dg_train = DataGenerator(env_name=env_name, seed=seed)

        # s_prime - future state after you taken the action from state s
        state_prime, state, action, reward = dg_train.get_samples(n_samples_train)

        # create training input pairs
        s_a_pairs = np.concatenate([state, action[:, np.newaxis]], axis=1)

        # solve regression problem s_prime = f(s,a)
        dynamics_model = SklearnModel(type)
        dynamics_model.fit(s_a_pairs, state_prime)

        # solve regression problem r = g(s,a)
        reward_model = SklearnModel(type)
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

    title_prefix = "Model: {} - {}\n".format(model_name, env_name)

    for i in range(dyn_history_test.shape[1]):
        plt.figure(i)
        plt.plot(data_point_range, dyn_history_test[:, i], label="Test")
        plt.plot(data_point_range, dyn_history_train[:, i], label="Train")
        plt.xlabel("# Samples")
        plt.ylabel("MSE")

        plt.title(title_prefix + "Dynamics Performance (State_{}) for different Sample Sizes".format(i))
        plt.legend()

    plt.figure(dyn_history_test.shape[1] + 1)
    plt.plot(data_point_range, rwd_history_test, label="Test")
    plt.plot(data_point_range, rwd_history_train, label="Train")
    plt.xlabel("# Samples")
    plt.ylabel("MSE")
    plt.title(title_prefix + "Reward Performance for different Sample Sizes")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
