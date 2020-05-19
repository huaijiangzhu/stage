import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym import spaces
import pybullet as p
import numpy as np
import time
from tqdm import trange
from dotmap import DotMap

from stage.tasks.base import Task
from stage.costs.cost import Cost
from stage.controllers.trivial import Identity, OpenLoop

from stage.utils.nn import beye, bquad, flatten_non_batch
from stage.utils.jacobian import AutoDiff
from stage.utils.nn import renew

from stage.utils.robotics import ForwardKinematics
from stage.tasks.kuka.params import JOINT_XYZ, JOINT_RPY, JOINT_AXIS, LINK_XYZ

class KukaReaching(Task):
    env_name = "Kuka-v0"
    task_horizon = 150
    nq, nv, nu, nx = 7, 7, 7, 14
    goal = np.array([0.2, 0.5, 0.1])
    

    def __init__(self,
                 dt_control=0.01,
                 dt_env=0.001, 
                 render=False):

        self.cost = DefaultCost()
        super().__init__(dt_env, dt_control, self.cost, render)
        self.update_goal(self.goal, noise_std=0)

        self.q_ub = torch.Tensor(self.env.q_ub)
        self.q_lb = torch.Tensor(self.env.q_lb)

    def update_goal(self, goal, noise_std=0):
        if noise_std > 0:
            goal += np.random.normal(loc=0, scale=noise_std, size=(3))

        q_desired = p.calculateInverseKinematics(self.env.robot_id,
                                                 6,
                                                 goal)

        self.cost.desired = torch.cat([torch.Tensor(q_desired), torch.zeros(self.nq)])

    def act(self, x, controller, params, random):
        control_repetition = int(self.dt_control/self.dt_env)
        a = controller(x, params, random)
        x0 = x.clone()

        for i in range(control_repetition):
            u = torch.flatten(controller.actor(x, a))
            obs, reward, done, info = self.env.step(u)
            x = torch.Tensor(obs[:self.nx])

        dx = x - x0
        transition = torch.cat((x0, a, dx), dim=0)
        info.transition = transition
        return x, reward, done, info

    def perform(self, goal, controller):
        x = self.reset(goal, noise_std=0)
        controller.reset()
        start = time.time()

        data, log = self.unroll(x, controller) 
        end = time.time()

        ro = np.sum([log[i].obs_reward for i in range(self.task_horizon)])
        ra = np.sum([log[i].act_reward for i in range(self.task_horizon)])

        print ("avg. decision time: ", (end - start)/self.task_horizon)
        print ("obs. reward: ", ro)
        print ("act. reward: ", ra)

        return data, log

    def reset(self, goal=None, noise_std=0):
        super().reset()
        if goal is None:
            goal = self.goal
        self.update_goal(goal, noise_std)
        obs, _, _, _ = self.env.reset()
        x = torch.Tensor(obs[:self.nx])
        return x

class DefaultCost(Cost):
    def __init__(self):
        super().__init__()
        self.nq = 7
        self.nx = 14
        self.nu = 7
        self.desired = torch.zeros(self.nx)
        self.fwk = ForwardKinematics(self.nq, JOINT_XYZ, JOINT_RPY, JOINT_AXIS, LINK_XYZ)
        self.Q = torch.diag(torch.Tensor([1,1,1,1,1,1,1,0,0,0,0,0,0,0])).unsqueeze(0)
        self.R = 1e-6 * torch.eye(self.nu).unsqueeze(0)
        self.d = AutoDiff()

    def forward(self, x, u, t=0, terminal=False):

        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        if u.ndimension() == 1:
            u = u.unsqueeze(0) 

        cost = DotMap()

        cost.obs = self.obs_cost(x)
        if not terminal:
            cost.act = self.action_cost(u)
        else:
            cost.act = 0

        l = cost.obs + cost.act
        cost.l = l

        return cost

    # def obs_cost(self, x, t=0, terminal=False):
        
    #     x = x[:, :self.nx]
    #     diffx = x - self.desired
    #     Q = self.Q.expand(x.shape[0], *self.Q.shape[1:])
    #     return bquad(Q, diffx)

    def obs_cost(self, x, t=0, terminal=False):
        
        q = x[:, :self.nq]
        ee_pos = self.fwk(q, 1)[:, :3, 3]
        # obstacle_pos = torch.Tensor([1.0960, 0.0000, 1.1710])
        # diff_obstacle = ee_pos - obstacle_pos
        diff_goal = ee_pos - torch.Tensor([0.2, 0.5, 0.1])
        Q = beye(1, 3, 3)
        Q = Q.expand(x.shape[0], *Q.shape[1:])

        return bquad(Q, diff_goal)

    def action_cost(self, u, t=0, terminal=False):

        R = self.R.expand(u.shape[0], *self.R.shape[1:])
        return bquad(R, u)





    