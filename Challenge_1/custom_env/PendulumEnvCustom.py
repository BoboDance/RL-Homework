"""
@file: CustomPendulum.py
Created on 26.11.18
@project: RL-Homework
@author: queensgambit

Custom version of the Pendulum Environment which uses an angular output.
"""


from gym.envs.classic_control import PendulumEnv
import numpy as np


class PendulumEnvCustom(PendulumEnv):

    def __init__(self):
        super().__init__(self)

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([theta, thetadot])
