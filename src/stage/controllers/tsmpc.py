import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from stage.controllers.base import Controller
from stage.costs.tsmpc_cost import TSMPCCost
from stage.optimizers.cem import CEM
from stage.optimizers.cmaes import CMAES
from stage.utils.nn import truncated_normal

class TSMPC(nn.Module):

    def __init__(self, action_ub, action_lb, dynamics, step_cost,
                 plan_horizon, n_particles, pop_size, inner_loop_controller):
        super().__init__()
        self.nq, self.nv = dynamics.nq, dynamics.nv
        self.na = action_lb.shape[0]
        self.dynamics = dynamics
        self.action_ub, self.action_lb = action_ub, action_lb
        self.inner_loop_controller = inner_loop_controller

        self.plan_horizon, self.n_particles = plan_horizon, n_particles
        self.pop_size = pop_size

        self.optimizer = CMAES(na=self.na, horizon=self.plan_horizon, pop_size=self.pop_size,
                             upper_bound=self.action_ub.repeat(self.plan_horizon),
                             lower_bound=self.action_lb.repeat(self.plan_horizon))

        self.cost = TSMPCCost(self.plan_horizon, self.n_particles, self.pop_size,
                              self.dynamics, self.inner_loop_controller, self.na, step_cost)
        self.prev_sol = ((self.action_lb + self.action_ub)/2).repeat(self.plan_horizon)
        self.init_var = ((self.action_ub - self.action_lb) **2 / 16).repeat(self.plan_horizon)

    @torch.no_grad()
    def openloop(self, x, init_sol, horizon):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        self.cost.obs = x
        self.cost.horizon = horizon

        init_var = ((self.action_ub - self.action_lb) **2 / 16).repeat(horizon)
        self.optimizer.reset(sol_dim=horizon * self.na,
                             upper_bound=self.action_ub.repeat(horizon),
                             lower_bound=self.action_lb.repeat(horizon))
        sol, _, opt = self.optimizer(self.cost, init_sol, init_var)
        return sol, opt

    @torch.no_grad()
    def forward(self, x, params=None, random=False):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        self.cost.obs = x
        self.cost.horizon = self.plan_horizon

        if random:
            mean = (self.action_lb + self.action_ub)/2
            var =  ((self.action_ub - self.action_lb)**2)/16
            lb_dist, ub_dist = mean - self.action_lb, self.action_ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            a = truncated_normal(mean.shape, mean, torch.sqrt(constrained_var))

        else:
            sol, _, _ = self.optimizer(self.cost, self.prev_sol, self.init_var)
            self.prev_sol = torch.cat((sol[self.na:], (self.action_lb + self.action_ub)/2))
            a = sol[:self.na]

        return a

    def reset(self):
        self.optimizer.reset(sol_dim=self.plan_horizon * self.na,
                             upper_bound=self.action_ub.repeat(self.plan_horizon),
                             lower_bound=self.action_lb.repeat(self.plan_horizon))
        self.prev_sol = ((self.action_lb + self.action_ub)/2).repeat(self.plan_horizon)




