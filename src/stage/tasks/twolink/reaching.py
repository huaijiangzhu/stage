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
from stage.costs.step_cost import StepCost
from stage.controllers.trivial import Identity, OpenLoop

class TwoLinkReaching(Task):
    env_name = "TwoLink-v0"
    task_horizon = 100
    nq, nv, nu, nx = 2, 2, 2, 4
    goal = np.array([-0.5*np.pi, 0, 0, 0])
    

    def __init__(self,
                 dt_control=0.01,
                 dt_env=0.001, 
                 render=False):

        self.step_cost = TwoLinkStepCost()
        super().__init__(dt_env, dt_control, self.step_cost, render)
        self.update_goal(self.goal, noise=False)

        self.q_ub = torch.Tensor([3.2, 3.2])
        self.q_lb = torch.Tensor([-3.2, -3.2])

    def update_goal(self, goal, noise=False):
        if noise:
            goal += np.random.normal(loc=0, scale=0.1, size=(self.nx))
        
        self.step_cost.desired = torch.Tensor(goal)

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
        obs, _, _, _ = self.env.reset()
        x = torch.Tensor(obs[:self.nx])
        return x


class TwoLinkStepCost(StepCost):
    def __init__(self):
        super().__init__()
        self.desired = torch.zeros(4)

    def obs_cost(self, obs):
        if obs.ndimension() == 1:
            obs = obs.unsqueeze(0)
        q = obs[:, :2]
        v = obs[:, 2:4]
        diff_q = q - self.desired[:2]
        return torch.sum(diff_q**2, dim=1) 

    def action_cost(self, action):
        if action.ndimension() == 1:
            action = action.unsqueeze(0)
        return 1e-5 * (action ** 2).sum(dim=1)




    