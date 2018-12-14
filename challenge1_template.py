"""
Submission template for Programming Challenge 1: Dynamic Programming.
"""
from Challenge_1.Models.NNModelPendulum import NNModelPendulum
from Challenge_1.Models.NNModelQube import NNModelQube
from Challenge_1.util.state_preprocessing import reconvert_state_to_angle, normalize_input, get_feature_space_boundaries, convert_state_to_sin_cos
from Challenge_1.nn_training import train, create_dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import logging

info = dict(
    group_number=16,  # change if you are an existing seminar/project group
    authors="Fabian Otto; Johannes Czech;",
    description="We train two neural network for both the dynamics and reward prediction."
                "In order to get lower MSE for our models we replaced the angle by sin()/cos() of "
                "the angle in Pendulum-v2."
                "Furthermore we linear scale our input and target features to the range [0,1]."
                "Pendulum v-0 has two features for the angle. That's why we convert it by using atan2() for "
                "policy and value iteration.")


def get_model(env, max_num_samples):
    """
    Sample up to max_num_samples transitions (s, a, s', r) from env
    and fit a parametric model s', r = f(s, a).

    :param env: gym.Env
    :param max_num_samples: maximum number of calls to env.step(a)
    :return: function f: s, a -> s', r
    """

    logging.info('Start Training Models which emulate the Environment')

    ## Settings
    env_name = 'Pendulum-v0'
    lr = 1e-3
    optimizer_name = 'rmsprop'
    export_plots = False
    export_models = False
    seed = 1234
    batch_size_dynamics = 64
    batch_size_reward = 256
    n_samples = max_num_samples
    n_epochs = 20 #150

    # index list of angle features
    if env_name == 'Pendulum-v0':
        angle_features = [0]
        convert_to_sincos = False
    elif env_name == 'Pendulum-v2':
        convert_to_sincos = True

    # define the input- and output shape of the NN
    n_inputs = env.observation_space.shape[0] + env.action_space.shape[0]
    n_outputs = env.observation_space.shape[0]

    if convert_to_sincos is True:
        # if our source stated doesn't have sin cos features we must add them by additional parameters
        n_inputs += len(angle_features)
        n_outputs += len(angle_features)

    X_low, X_high = get_feature_space_boundaries(env, []) #angle_features)

    # scaling defines how our outputs will be scaled after the tanh function
    # for this we use all state features ergo all of X_high excluding the last action feature
    scaling = X_high[:-1]

    # here it's assumed to only work with the pendulum
    dynamics_model = NNModelPendulum(n_inputs=n_inputs,
                                     n_outputs=n_outputs,
                                     scaling=scaling, optimizer='adam')

    reward_model = NNModelPendulum(n_inputs=n_inputs,
                                   n_outputs=1,
                                   scaling=None, optimizer='adam')

    lossfunction = nn.MSELoss()

    if optimizer_name == 'rmsprop':
        optimizer_dynamics = optim.RMSprop(dynamics_model.parameters(), lr=lr)
        optimizer_reward = optim.RMSprop(reward_model.parameters(), lr=lr)
    elif optimizer_name == 'adam':
        optimizer_dynamics = optim.Adam(dynamics_model.parameters(), lr=lr)
        optimizer_reward = optim.Adam(reward_model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer_dynamics = optim.SGD(dynamics_model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        optimizer_reward = optim.SGD(reward_model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    else:
        raise Exception('Unsupported optimizer')

    # Create Datasets for Training and Testing
    s_a_pairs_train, state_prime_train, reward_train = create_dataset(env, env_name, seed, n_samples, angle_features,
                                                                      convert_to_sincos)
    s_a_pairs_test, state_prime_test, reward_test = create_dataset(env, env_name, seed + 1, n_samples, angle_features,
                                                                   convert_to_sincos)

    # Normalize the input X for the neural network
    s_a_pairs_train = normalize_input(s_a_pairs_train, X_low, X_high)
    s_a_pairs_test = normalize_input(s_a_pairs_test, X_low, X_high)

    state_prime_train = normalize_input(state_prime_train, X_low[:-1], X_high[:-1])
    state_prime_test = normalize_input(state_prime_test, X_low[:-1], X_high[:-1])

    # Start the training process
    # --------------------------
    # Train the Dynamics Model
    for model_type, model, optimizer, X, Y, X_val, Y_val, batch_size in zip(['dynamics', 'reward'],
                                                                [dynamics_model, reward_model],
                                                                [optimizer_dynamics, optimizer_reward],
                                                                [s_a_pairs_train, s_a_pairs_train],
                                                                [state_prime_train, reward_train],
                                                                [s_a_pairs_test, s_a_pairs_test],
                                                                [state_prime_test, reward_test],
                                                                [batch_size_dynamics, batch_size_reward]):

        train_loss, val_loss = train(dynamics_model, optimizer=optimizer_dynamics,
                                                   X=s_a_pairs_train, Y=state_prime_train, X_val=s_a_pairs_test,
                                                   Y_val=state_prime_test, batch_size=batch_size_dynamics,
                                                   n_epochs=n_epochs, lossfunction=lossfunction)
        # Visualize the training process
        plt.title('%s: Learning Rewards\n Batch-Size=%d, lr=%f, optimizer=%s' %
                  (env_name, batch_size, lr, optimizer_name))
        plt.plot(train_loss, label='train-loss')
        plt.plot(val_loss, label='val-loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.show()

        if export_plots is True:
            plt.savefig('Plots/%s_Reward.png' % env_name)

        print()
        logging.info('Learning %s:' % model_type.upper())
        logging.info('Final Train MSE after %d epochs: %.5f' % (n_epochs, train_loss[-1]))
        logging.info('Final Validation MSE after %d epochs: %.5f' % (n_epochs, val_loss[-1]))

        if export_models is True:
            # Save the weights of the trained model
            export_name = "./Challenge_1/Weights/model_%s_%s_mse_%.8f.params" % (model_type, env_name, val_loss[-1])
            torch.save(reward_model.state_dict(), export_name)
            logging.info('Your weights have been saved to %s successfully!' % export_name)

    return lambda obs, act: (2*obs + act, obs@obs + act**2)


def get_policy(model, observation_space, action_space):
    """
    Perform dynamic programming and return the optimal policy.

    :param model: function f: s, a -> s', r
    :param observation_space: gym.Space
    :param action_space: gym.Space
    :return: function pi: s -> a
    """
    return lambda obs: action_space.high
