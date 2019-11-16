from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os

import numpy as np
import gym
from gym import utils
import pybullet as p
from dotmap import DotMap
from stage.envs.base import BaseEnv
import torch

class TwoLinkEnv(BaseEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}
    nq, nv, nu, nx = 2, 2, 2, 4

    def __init__(self, dt, step_cost=None, do_render=False):
        super().__init__(dt, step_cost, do_render)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.robot_urdf = os.path.join(dir_path, 'urdf/twolink.urdf')
        self.action_space = gym.spaces.Box(-50, 50,(self.nu,))
        self.reset()

    def step(self, tau=None):
        if tau is None:
            tau = np.zeros(self.nj)
        if not isinstance(tau, np.ndarray):
            tau = tau.detach().cpu().numpy()

        zero_gains = np.zeros(self.nj)
        p.setJointMotorControlArray(self.robot_id, 
                                    self.bullet_joint_ids, p.TORQUE_CONTROL,
                                    forces=tau,
                                    positionGains=zero_gains, velocityGains=zero_gains)
        p.stepSimulation()
        q, v = self.get_state()
        q = self.wrap(q)
        obs = np.concatenate((q, v))
        if self.step_cost is None:
            obs_reward = 0
            act_reward = 0
        else:
            obs_cost, act_cost = self.step_cost(torch.Tensor(obs), torch.Tensor(tau))
            obs_reward = -obs_cost.detach().cpu().numpy()
            act_reward = -act_cost.detach().cpu().numpy()

        reward = obs_reward + act_reward
        info = DotMap()
        info.obs_reward = obs_reward
        info.act_reward = act_reward
        info.reward = reward
        done = False
        return obs, reward, done, info

    def wrap(self, q):
        q = np.arctan2(np.sin(q), np.cos(q))
        return q

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        cam_dist = 1.3
        cam_yaw = 180
        cam_pitch = -40
        RENDER_HEIGHT = 720
        RENDER_WIDTH = 960

        base_pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                          distance=cam_dist,
                                                          yaw=cam_yaw,
                                                          pitch=cam_pitch,
                                                          roll=0,
                                                          upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=RENDER_WIDTH,
                                            height=RENDER_HEIGHT,
                                            viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix,
                                            renderer=p.ER_BULLET_HARDWARE_OPENGL)
        #renderer=self._p.ER_TINY_RENDERER)

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def get_state(self):
        q = np.zeros(self.nq)
        v = np.zeros(self.nv)

        # Query the joint readings.
        joint_states = p.getJointStates(self.robot_id, self.bullet_joint_ids)

        for i in range(self.nv):
            q[i] = joint_states[i][0]
            v[i] = joint_states[i][1]

        return q, v

    def set_state(self, q, v):
        for i in range(self.nj):
            p.resetJointState(self.robot_id, i, q[i], v[i])

    def reset(self, init_state=None):
        p.resetSimulation()
        self.robot_id = self.load_body(self.robot_urdf, use_fixed_base=True)
        super().reset(init_state)
        return self.step()

    def close(self):
        p.disconnect()


