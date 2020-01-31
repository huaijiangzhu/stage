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

from stage.utils.nn import bquad, flatten_non_batch
from stage.utils.jacobian import AutoDiff
from stage.utils.nn import renew

class TwoLinkReaching(Task):
    env_name = "TwoLink-v0"
    task_horizon = 100
    nq, nv, nu, nx = 2, 2, 2, 4
    goal = np.array([0, 0, 0, 0])
    

    def __init__(self, 
                 dt_control=0.01,
                 dt_env=0.001, 
                 render=False):

        self.cost = DefaultCost()
        super().__init__(dt_env, dt_control, self.cost, render)
        self.update_goal(self.goal, noise=False)

        self.q_ub = torch.Tensor([3.2, 3.2])
        self.q_lb = torch.Tensor([-3.2, -3.2])

    def update_goal(self, goal, noise=False):
        if noise:
            goal += np.random.normal(loc=0, scale=0.1, size=(self.nx))
        
        self.cost.desired = torch.Tensor(goal)

    def perform(self, goal, controller):
        x = self.reset(goal, noise=False)
        start = time.time()

        data, log = self.unroll(x, controller) 
        end = time.time()

        ro = np.sum([log[i].obs_reward for i in range(self.task_horizon)])
        ra = np.sum([log[i].act_reward for i in range(self.task_horizon)])

        print ("avg. decision time: ", (end - start)/self.task_horizon)
        print ("obs. reward: ", ro)
        print ("act. reward: ", ra)

        return data, log

    def reset(self, goal=None, noise=False):
        super().reset()
        if goal is None:
            goal = self.goal
        self.update_goal(goal, noise)
        q = np.array([0.5*np.pi, 0])
        v = np.zeros(self.nv)
        obs, _, _, _ = self.env.reset((q, v))
        x = torch.Tensor(obs[:self.nx])
        return x


class DefaultCost(Cost):
    def __init__(self):
        super().__init__()
        self.nx = 4
        self.nu = 2
        self.desired = torch.zeros(self.nx)
        self.Q = torch.diag(torch.Tensor([1,1,0,0])).unsqueeze(0)
        self.R = 1e-5 * torch.eye(self.nu).unsqueeze(0)
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

    def obs_cost(self, x, t=0, terminal=False):
        
        x = x[:, :self.nx]
        diffx = x - self.desired
        Q = self.Q.expand(x.shape[0], *self.Q.shape[1:])

        return bquad(Q, diffx)

    def action_cost(self, u, t=0, terminal=False):

        R = self.R.expand(u.shape[0], *self.R.shape[1:])
        return bquad(R, u)

    ### for DDP

    def l(self, x, a, t=0, terminal=False, diff=False):
        x, a = renew(x), renew(a)

        if diff:
            x.requires_grad = True
            a.requires_grad = True

        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        if a.ndimension() == 1:
            a = a.unsqueeze(0) 

        u = self.actor(x, a)
        cost = self.forward(x, u, t, terminal)

        if diff:
            cost.lx = flatten_non_batch(self.lx(cost.l, x, a, t))
            cost.lxx = self.lxx(cost.lx, x, a, t)

            if not terminal:
                cost.la = flatten_non_batch(self.la(cost.l, x, a, t))
                cost.lax = self.lax(cost.la, x, a, t)
                cost.laa = self.laa(cost.la, x, a, t)

        return cost

    def lx(self, l, x, a, t):
        return self.d(l, x)

    def lxx(self, lx, x, a, t):
        return self.d(lx, x)

    def la(self, l, x, a, t):
        return self.d(l, a)

    def laa(self, la, x, a, t):
        return self.d(la, a)

    def lax(self, la, x, a, t):
        return self.d(la, x)




    