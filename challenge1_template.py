"""
Submission template for Programming Challenge 1: Dynamic Programming.
"""
from Challenge_1.Models.NNModelPendulum import NNModelPendulum
from Challenge_1.Models.NNModelQube import NNModelQube
from Challenge_1.util.state_preprocessing import reconvert_state_to_angle, normalize_input, get_feature_space_boundaries, convert_state_to_sin_cos, unnormalize_input
from Challenge_1.nn_training import train, create_dataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch
import logging
import numpy as np

info = dict(
    group_number=16,  # change if you are an existing seminar/project group
    authors="Fabian Otto; Johannes Czech;",
    description="We train two neural network for both the dynamics and reward prediction."
                "In order to get lower MSE for our models we replaced the angle by sin()/cos() of "
                "the angle in Pendulum-v2."
                "Furthermore we linear scale our input and target features to the range [0,1]."
                "Pendulum v-0 has two features for the angle. That's why we convert it by using atan2() for "
                "policy and value iteration.")

# Global variables
# (used in order to be compatible with the template an provide a cleaner inference method
x_low = None
x_high = None
dynamics_model = None
reward_model = None
convert_to_sincos = None
angle_features = None


def get_model(env, max_num_samples):
    """
    Sample up to max_num_samples transitions (s, a, s', r) from env
    and fit a parametric model s', r = f(s, a).

    :param env: gym.Env
    :param max_num_samples: maximum number of calls to env.step(a)
    :return: function f: s, a -> s', r
    """

    global x_low
    global x_high
    global dynamics_model
    global reward_model
    global convert_to_sincos
    global angle_features

    logging.info('Start Training Models which emulate the Environment')

    # Settings
    env_name = 'Pendulum-v0'
    lr = 1e-3
    optimizer_name = 'rmsprop'
    export_plots = False
    export_models = False
    seed = 1234
    batch_size_dynamics = 64
    batch_size_reward = 256
    n_samples = max_num_samples
    n_epochs = 150

    # index list of angle features
    if env_name == 'Pendulum-v0':
        angle_features = []
        convert_to_sincos = False
    elif env_name == 'Pendulum-v2':
        angle_features = [0]
        convert_to_sincos = True

    # define the input- and output shape of the NN
    n_inputs = env.observation_space.shape[0] + env.action_space.shape[0]
    n_outputs = env.observation_space.shape[0]

    if convert_to_sincos is True:
        # if our source stated doesn't have sin cos features we must add them by additional parameters
        n_inputs += len(angle_features)
        n_outputs += len(angle_features)

    x_low, x_high = get_feature_space_boundaries(env, angle_features)

    # scaling defines how our outputs will be scaled after the tanh function
    # for this we use all state features ergo all of X_high excluding the last action feature
    scaling = x_high[:-1]

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
    s_a_pairs_train = normalize_input(s_a_pairs_train, x_low, x_high)
    s_a_pairs_test = normalize_input(s_a_pairs_test, x_low, x_high)

    state_prime_train = normalize_input(state_prime_train, x_low[:-1], x_high[:-1])
    state_prime_test = normalize_input(state_prime_test, x_low[:-1], x_high[:-1])

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

        train_loss, val_loss = train(model, optimizer=optimizer, X=X, Y=Y, X_val=X_val, Y_val=Y_val,
                                     batch_size=batch_size_dynamics, n_epochs=n_epochs, lossfunction=lossfunction)

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
        logging.info('Final Train MSE after %d epochs: %.5f (on normalized targets)' % (n_epochs, train_loss[-1]))
        logging.info('Final Validation MSE after %d epochs: %.5f (on normalized targets)' % (n_epochs, val_loss[-1]))

        if export_models is True:
            # Save the weights of the trained model
            export_name = "./Challenge_1/Weights/model_%s_%s_mse_%.8f.params" % (model_type, env_name, val_loss[-1])
            torch.save(reward_model.state_dict(), export_name)
            logging.info('Your weights have been saved to %s successfully!' % export_name)

    # return the the inference interface to the dynamics and reward model
    return provide_inference


def provide_inference(obs, act):
    """
    Provides the function f: s, a -> s', r

    :param obs: Observation of the state in numpy matrix form
    :param act: Action which the agent is taking or wants to take
    :return: State prime / predicted state (next state after applying the action)
             Predicted reward after applying the action
    """

    s_a = np.concatenate([obs, act], axis=0)

    # normalize the state action pair to the range [0,1]
    s_a = normalize_input(s_a, x_low, x_high)

    # request the next state from our dynamics model
    state_prime_pred = dynamics_model.predict(s_a)

    # after predicting the state prime from our dynamic model we should rennormalize it back
    # to it's original from. The sin(angle) replaces the original angle and the cos(angle) is appended to the
    # original state representation
    # unnormalize the state back to it's original state ranges
    state_prime_pred = unnormalize_input(state_prime_pred, x_low[:-1], x_high[:-1])
    # reconvert the angle feature back to a single angle to have a more compact representation
    if convert_to_sincos is True:
        state_prime_pred = reconvert_state_to_angle(state_prime_pred, angle_features)

    # request the reward prediction of the corresponding reward from our model
    reward = reward_model.predict(s_a)

    return state_prime_pred, reward


def get_policy(model, observation_space, action_space):
    """
    Perform dynamic programming and return the optimal policy.

    :param model: function f: s, a -> s', r
    :param observation_space: gym.Space
    :param action_space: gym.Space
    :return: function pi: s -> a
    """
    return lambda obs: action_space.high
