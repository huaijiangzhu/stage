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

from stage.controllers.trivial import Identity, OpenLoop
from stage.controllers.pd import PD, PDFull

from stage.controllers.tsmpc import TSMPC
from stage.optimizers.cem import CEM
from stage.costs.tsmpc_cost import TSMPCCost
from stage.costs.step_cost import StepCost

from stage.dynamics.probabilistic_ensemble import ProbabilisticEnsemble, Dx
from stage.utils.nn import swish, get_affine_params, truncated_normal

class TwoLinkPETS(Task):
    env_name = "TwoLink-v0"
    task_horizon = 100
    train_iterations = 20
    rollouts_per_iteration = 1
    plan_horizon = 25
    n_particles = 20
    pop_size = 400
    ensemble_size = 5
    nn_epochs = 10
    nq, nv, nu, nx = 2, 2, 2, 4
    goal = np.array([-0.5*np.pi, 0, 0, 0])
    

    def __init__(self,
                 dt_control=0.01,
                 dt_env=0.001, 
                 render=False, 
                 dynamics_path=None, 
                 action_parameterization='pd', 
                 learn_closed_loop_dynamics=True):

        assert self.n_particles % self.ensemble_size == 0
        self.step_cost = TwoLinkStepCost()
        self.learn_closed_loop_dynamics = learn_closed_loop_dynamics
        self.dx = TwoLinkDx
        self.update_goal(self.goal, noise=False)
        super().__init__(dt_env, dt_control, self.step_cost, render)

        goal_ub = torch.Tensor([3.2, 3.2])
        goal_lb = torch.Tensor([-3.2, -3.2])

        if action_parameterization=='pd':
            self.na = 4
            gain_ub = 50 * torch.ones((self.nq))
            gain_lb = 0.01 * torch.ones((self.nq))

            self.action_ub = torch.cat((gain_ub, goal_ub))
            self.action_lb = torch.cat((gain_lb, goal_lb))
            self.inner_loop_controller = PD

        elif action_parameterization=='pd_full':
            self.na = 6
            
            kp_ub = 50 * torch.ones((self.nq))
            kp_lb = 0.01 * torch.ones((self.nq))

            kd_ub = 20 * torch.ones((self.nq))
            kd_lb = 0.001 * torch.ones((self.nq))
            
            self.action_ub = torch.cat((kp_ub, kd_ub, goal_ub))
            self.action_lb = torch.cat((kp_lb, kd_lb, goal_lb))
            self.inner_loop_controller = PDFull

        elif action_parameterization=='torque':

            self.na = 2
            self.action_ub = torch.Tensor(self.env.action_space.high)
            self.action_lb = torch.Tensor(self.env.action_space.low)
            self.inner_loop_controller = Identity

        self.load_dynamics(dynamics_path)
        self.controller = TSMPC(self.action_ub, self.action_lb,
                                self.dynamics, self.step_cost,
                                self.plan_horizon, self.n_particles, self.pop_size,
                                self.inner_loop_controller(self.nq, self.nv, self.nu))
        

    def update_goal(self, goal, noise=False):
        if noise:
            goal += np.random.normal(loc=0, scale=0.1, size=(self.nx))
        
        self.step_cost.desired = torch.Tensor(goal)

    def load_dynamics(self, path):
        if self.learn_closed_loop_dynamics:
            self.dynamics = ProbabilisticEnsemble(self.nq, self.nv, self.na, self.dt_control, 
                                                  self.dx, self.ensemble_size, 
                                                  learn_closed_loop_dynamics=True)
        else:
            self.dynamics = ProbabilisticEnsemble(self.nq, self.nv, self.nu, self.dt_control, 
                                                  self.dx, self.ensemble_size)

        if path is not None:
            self.dynamics.dx.load_state_dict(torch.load(path))

    def save_dynamics(self, path):
        torch.save(self.dynamics.dx.state_dict(), path) 

    def learn(self, iteration, verbose=False):
        logs = []

        for i in range(iteration):
            
            x = self.reset(self.goal, noise=False)
            start = time.time()
            if self.data_train is None:
                self.data_train, log = self.unroll(x, random=True)
                end = time.time()
                self.dynamics.learn(self.data_train, self.nn_epochs, verbose=verbose)
            else:
                new_data, log = self.unroll(x, random=False)
                end = time.time()
                self.data_train = torch.cat((self.data_train, new_data), dim=0)
                self.dynamics.learn(self.data_train, self.nn_epochs, verbose=verbose)
            
            if verbose:
                print ('Iteration: ', i)
                ro = np.sum([log[i].obs_reward for i in range(self.task_horizon)])
                ra = np.sum([log[i].act_reward for i in range(self.task_horizon)])
                print ("avg. decision time: ", (end - start)/self.task_horizon)
                print ("obs. reward: ", ro)
                print ("act. reward: ", ra)
                
            logs.append(log)
        return logs

    def act(self, x, controller, params, random):
        control_repetition = int(self.dt_control/self.dt_env)
        a = controller(x, params, random)
        x0 = x.clone()

        for i in range(control_repetition):
            u = torch.flatten(controller.inner_loop_controller(x, a))
            obs, reward, done, info = self.env.step(u)
            x = torch.Tensor(obs[:self.nx])

        dx = x - x0

        if self.learn_closed_loop_dynamics:
            transition = torch.cat((x0, a, dx), dim=0)
        else:
            transition = torch.cat((x0, u, dx), dim=0)

        info.transition = transition
        return x, reward, done, info

    def perform(self, goal, action_sequence=None):
        x = self.reset(goal, noise=False)
        start = time.time()
        if action_sequence is not None:
            openloop_controller = OpenLoop(self.nq, self.nv, self.nu, self.na, action_sequence)
            openloop_controller.inner_loop_controller = self.controller.inner_loop_controller
            params_generator = lambda n : n
            data, log = self.unroll(x, openloop_controller, params_generator)
        else:
            data, log = self.unroll(x) 
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
        self.controller.reset()
        obs, _, _, _ = self.env.reset()
        x = torch.Tensor(obs[:self.nx])
        return x

class TwoLinkDx(Dx):
    def __init__(self, ensemble_size, nx, na):
        super().__init__(ensemble_size, nx, na)
        self.lin0_w, self.lin0_b = get_affine_params(self.ensemble_size, self.nin, 200)
        self.lin1_w, self.lin1_b = get_affine_params(self.ensemble_size, 200, 200)
        self.lin2_w, self.lin2_b = get_affine_params(self.ensemble_size, 200, 200)
        self.lin3_w, self.lin3_b = get_affine_params(self.ensemble_size, 200, self.nout)
        
    def forward(self, inputs, return_logvar=False):

        inputs = (inputs - self.inputs_mu) / self.inputs_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b

        mean = inputs[:, :, :self.nx]
        logvar = inputs[:, :, self.nx:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if return_logvar:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def compute_decays(self):

        lin0_decays = 0.000025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.00005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.000075 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.000075 * (self.lin3_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays


class TwoLinkStepCost(StepCost):
    def __init__(self):
        super().__init__()

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




    