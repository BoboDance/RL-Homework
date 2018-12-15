"""
@file: state_preprocessing.py
Created on 09.12.18
@project: RL-Homework

Collection of methods which are used to convert the state representation of the angle.
"""

import numpy as np


def convert_state_to_sin_cos(state, angle_features):
    """
    Replaces the angles features of a state representation by two features sin(angle) and cos(angle).
    It's possible to give multiple states in one array.

    :param state: State representation with angle features in radiant
    :param angle_features: List of the indices which correspond to the column of the angle features
    :return: Corresponding numpy matrix of the state in which the angle was replaced by sin(angle), cos(angle)
    """

    if not angle_features:
        return state

    state = np.atleast_2d(state)

    angles = state[:, angle_features]
    # replace the angle feature with the sin of the angle
    state[:, angle_features] = np.sin(angles)
    # insert cosine of the angle in front
    return np.insert(state, angle_features, np.cos(angles), axis=1)


def reconvert_state_to_angle(state, angle_features):
    """
    Converts the state representation with sin(angle), cos(angle) features back to the original representation in which
    the angle is only one feature.

    :param state: State representation with sin(angle), cos(angle) feature/s
    :param angle_features: List of the indices which correspond to the column of the angle features
    :return: Corresponding numpy matrix in which sin(angle), cos(angle) was replaced by the original angle using the
            atan2(y, x)-function
    """

    if not angle_features:
        return state

    state = np.atleast_2d(state)

    cos = np.array([idx + i for i, idx in enumerate(angle_features)])
    sin = cos + 1
    sin_cos = np.concatenate([sin, cos])

    state[:, sin_cos] = np.clip(state[:, sin_cos], -1, 1)

    angle = np.arctan2(state[:, sin], state[:, cos])
    state[:, sin] = angle
    return np.delete(state, cos, axis=1)


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


def get_feature_space_boundaries(observation_space, action_space, angle_features):
    """
    Returns the boundaries for the environment features space.
    This is used for the state_action pair and assumes sin()/cos() of the angle
    :param observation_space: OpenAI-gym environment observation_space
    :param action_space: OpenAI-gym environment action_space
    :param angle_features: List of the features which are the angles in the original representation
    :return: x_low: Lower boundary of the feature space
            x_high: Upper boundary of the feature space
    """
    # define the vectors which describe the maximum and minimum of the feature space
    x_low = np.concatenate([observation_space.low, action_space.low])
    x_high = np.concatenate([observation_space.high, action_space.high])

    cos_low = [-1] * len(angle_features)
    cos_high = [1] * len(angle_features)

    x_low = np.insert(x_low, angle_features, np.array(cos_low))
    x_high = np.insert(x_high, angle_features, np.array(cos_high))

    return x_low, x_high
