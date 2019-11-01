from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import gym
from gym import utils
import pybullet as p
from dotmap import DotMap

class BaseEnv(gym.Env):

    def __init__(self, dt, step_cost, do_render=False):
        self.viewer = None
        self.do_render = do_render
        self.dt = dt
        self.step_cost = step_cost
        if self.do_render:
            cid = p.connect(p.GUI)
        else:
            cid = p.connect(p.DIRECT)

    def step(self, tau=None):
        raise NotImplementedError

    def get_state(self):
        q = np.zeros(self.nq)
        v = np.zeros(self.nv)

        # Query the joint readings.
        joint_states = p.getJointStates(self.robot_id, self.bullet_joint_ids)

        for i in range(self.nv):
            q[i] = joint_states[i][0]
            v[i] = joint_states[i][1]

        return q, v

    def load_body(self, body_urdf, use_fixed_base=True):
        body_id = p.loadURDF(body_urdf, useFixedBase=use_fixed_base)
        return body_id

    def initialize_robot(self, init_state=None):
        self.q_ub = np.zeros(self.nj)
        self.q_lb = np.zeros(self.nj)

        for i in range(self.nj):
            self.q_lb[i], self.q_ub[i] = p.getJointInfo(self.robot_id, i)[8:10]

        p.setJointMotorControlArray(self.robot_id, 
                            self.bullet_joint_ids, 
                            p.VELOCITY_CONTROL, 
                            forces=np.zeros(self.nu))
        
        q = np.zeros(self.nq)
        v = np.zeros(self.nv)
        if init_state is not None:
            q, v = init_state

        ## TODO take care of floating base system
        for i in range(self.nj):
            p.resetJointState(self.robot_id, i, q[i], v[i])

    def reset(self, init_state=None):
        p.setTimeStep(self.dt)
        p.setGravity(0, 0, 0)

        ## TODO take care of floating base system

        self.nj = p.getNumJoints(self.robot_id)
        self.bullet_joint_ids = np.arange(self.nj)
        self.initialize_robot(init_state)

        return self.get_state()

    def close(self):
        p.disconnect()


