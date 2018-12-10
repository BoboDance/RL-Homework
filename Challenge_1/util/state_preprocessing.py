"""
@file: state_preprocessing.py
Created on 09.12.18
@project: RL-Homework

Collection of methods which are used to convert the state representation of the angle.
"""

import numpy as np


def convert_state_to_sin_cos(state_angle, angle_features):
    """
    Replaces the angles features of a state representation by two features sin(angle) and cos(angle).
    It's possible to give multiple states in one array.

    :param state_angle: State representation with angle features in radiant
    :param angle_features: List of the indices which correspond to the column of the angle features
    :return: Corrsponding Numpy Matrix of the state in which the angle was replaced by sin(angle), cos(angle)
    """
    cos_feature_columns = np.zeros((len(state_angle), len(angle_features)))

    state_ret = np.array(state_angle)
    for i, angle_idx in enumerate(angle_features):
        # replace the angle feature with the sin of the angle
        state_ret[:, angle_idx] = np.sin(state_angle[:, angle_idx])
        # create a new column with the cosine of the angle
        cos_feature_columns[:, i] = np.cos(state_angle[:, angle_idx])

    state_ret = np.concatenate((state_ret, cos_feature_columns), axis=1)

    return state_ret


def reconvert_state_to_angle(state_sincos, angle_features):
    """
    Converts the state representation with sin(angle), cos(angle) features back to the original representation in which
    the angle is only one feature.

    :param state_sincos: State representation with sin(angle), cos(angle) feature/s
    :param angle_features: List of the indices which correspond to the column of the angle features
    :return: Corresponding numpy matrix in which sin(angle), cos(angle) was replaced by the original angle using the
            atan2(y, x)-function
    """

    state_ret = np.array(state_sincos)

    for i, angle_idx in enumerate(angle_features):
        # replace sin() and cos() feature by the single angle value again
        state_ret[:, angle_idx] = np.arctan2(state_sincos[:, angle_idx], state_sincos[:, state_sincos.shape[1]-len(angle_features) + angle_idx])

    state_ret = state_ret[:, :-len(angle_features)]

    return state_ret


def normalize_input(x, x_low, x_high):
    """
    Normalizes the state repersentation to the range [0,1] using linear scaling the boundaries for x.

    :param x: Feature space representation with arbitrary entries
    :param x_low: Vector indicating the lowest entry for each column
    :param x_high: Vector describing the highest possilbe entry for each feature
    :return:
    """
    x = x - x_low
    x /= (x_high - x_low)

    return x


def unnormalize_input(x, x_low, x_high):
    """
    Unormalizes the space back to the original state space dimension

    :param x: Feature space representation with entries between [0,1]
    :param x_low: Vector indicating the lowest entry for each column
    :param x_high: Vector describing the highest possilbe entry for each feature
    :return:
    """
    x *= (x_high - x_low)
    x = x + x_low

    return x


def get_feature_space_boundaries(env, angle_features):
    """
    Returns the boundaries for the environment features space.
    This is used for the state_action pair and assumes sin()/cos() of the angle
    :param env: OpenAI-gym environment handle
    :param angle_features: List of the features which are the angles in the original representation
    :return: x_low: Lower boundary of the feature space
            x_high: Uppter boundary of the feature space
    """
    # define the vectors which describe the maximum and minimum of the feature space
    x_low = np.concatenate([env.observation_space.low, len(angle_features) * [-1], env.action_space.low])
    x_high = np.concatenate(
        [env.observation_space.high, len(angle_features) * [1], env.action_space.high])
    for angle_idx in angle_features:
        x_low[angle_idx] = -1
        x_high[angle_idx] = 1

    return x_low, x_high
