import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from stage.controllers.base import Controller
from stage.optimizers.cem import CEM
from stage.utils.nn import truncated_normal

import pdb
class MSCEM(nn.Module):

    def __init__(self, dynamics, cost, actor,
                 plan_horizon, n_particles, pop_size):
        super().__init__()
        self.nq, self.nv, self.nx = dynamics.nq, dynamics.nv, dynamics.nx
        self.actor = actor
        self.dynamics = dynamics

        self.na = self.actor.na
        self.nxa = self.nx + self.na 

        ### construct solution bounds
        self.action_ub, self.action_lb = self.actor.action_ub, self.actor.action_lb
        self.state_ub, self.state_lb = self.dynamics.state_ub, self.dynamics.state_lb
        self.sol_lb = torch.cat((self.state_lb, self.action_lb))
        self.sol_ub = torch.cat((self.state_ub, self.action_ub))
        
        self.plan_horizon, self.n_particles = plan_horizon, n_particles
        self.pop_size = pop_size

        self.optimizer = CEM(nsol=self.plan_horizon*self.nxa, pop_size=self.pop_size,
                             ub=self.sol_ub.repeat(self.plan_horizon),
                             lb=self.sol_lb.repeat(self.plan_horizon))

        self.cost = MSCEMCost(self.plan_horizon, self.n_particles, self.pop_size,
                              self.dynamics, self.actor, cost)
        self.init_var = ((self.sol_ub - self.sol_lb) **2 / 16).repeat(self.plan_horizon)
        self.reset()

    @torch.no_grad()
    def openloop(self, x, init_sol, horizon):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        self.cost.obs = x
        self.cost.horizon = horizon

        init_var = ((self.sol_ub - self.sol_lb) **2 / 16).repeat(horizon)
        self.optimizer.reset(nsol=horizon*self.nxa,
                             ub=self.sol_ub.repeat(horizon),
                             lb=self.sol_lb.repeat(horizon))
        self.optimizer.horizon = horizon
        sol, _, opt = self.optimizer(self.cost, init_sol, init_var)
        return sol

    @torch.no_grad()
    def forward(self, x, params=None, random=False):
        if random:
            a = self.actor.sample()

        else:
            sol = self.openloop(x, self.prev_sol, self.plan_horizon)
            self.prev_sol = torch.cat((sol[self.nxa:], (self.sol_lb + self.sol_ub)/2))
            a = sol[self.nx:self.nxa]

        return a

    def regularize(self, ns):
        self.cost.ns = ns

    def reset(self):
        self.optimizer.reset(nsol=self.plan_horizon*self.nxa,
                             ub=self.sol_ub.repeat(self.plan_horizon),
                             lb=self.sol_lb.repeat(self.plan_horizon))
        self.optimizer.horizon = self.plan_horizon
        self.prev_sol = ((self.sol_lb + self.sol_ub)/2).repeat(self.plan_horizon)


class MSCEMCost(nn.Module):
    def __init__(self, horizon, n_particles, pop_size,
                 dynamics, actor, task_cost):
        super().__init__()
        self.dynamics = dynamics
        self.actor = actor
        self.nq, self.nv, self.nx = dynamics.nq, dynamics.nv, dynamics.nx
        self.na = actor.na
        self.nxa = self.nx + self.na
        self.ensemble_size = dynamics.ensemble_size
        self.horizon = horizon
        self.n_particles = n_particles
        self.pop_size = pop_size
        self.task_cost = task_cost
        self.shooting_interval_length = 30

        # number of samples for Lipschitz regularization
        self.ns = 0

    def forward(self, sol):
        sol = self.reshape_sol(sol)
        sol_x = sol[:, :, :self.nx]
        sol_a = sol[:, :, self.nx:]
        obs = self.reshape_obs(self.obs)
        sol_x[0, :, :] = obs # initial state is fixed

        
        H = self.horizon
        L = self.shooting_interval_length
        L = 1
        M = int(H / L)
        costs = torch.zeros(self.pop_size, H)

        for m in range(M):
            x = sol_x[m * L]
            for n in range(L):
                a = sol_a[m * L + n]
                prediction = self.dynamics.sample_predictions(x, a, self.n_particles)
                cost = self.task_cost.l(prediction.x, a).l
                cost = cost.view(-1, self.n_particles)
                costs[:, m * L + n] = cost.mean(dim=1)
                x = prediction.x
            
            # gap closing cost
            if m < M - 1:
                gap = x - sol_x[(m + 1) * L]
            else:
                gap = torch.zeros_like(x) # the last shooting interval does not have gap cost

            gap_cost = torch.exp(torch.norm(gap, p=2, dim=1))
            gap_cost = gap_cost.view(-1, self.n_particles)
            gap_cost -= torch.ones_like(gap_cost)
            costs[:, m * L + L - 1] += L * gap_cost.mean(dim=1)

        return costs

    def reshape_sol(self, sol):
        # action_traj : (pop_size, horizon*nsol)
        sol = sol.view(-1, self.horizon, self.nxa)
        # action_traj : (pop_size, horizon, nsol)
        transposed = sol.transpose(0, 1)
        # action_traj : (horizon, pop_size, nsol)
        expanded = transposed[:, :, None]
        # action_traj : (horizon, pop_size, 1, nsol)
        tiled = expanded.expand(-1, -1, self.n_particles, -1)
        # action_traj : (horizon, pop_size, n_particles, nsol)
        sol = tiled.contiguous().view(self.horizon, -1, self.nxa)
        # action_traj : (horizon, pop_size * n_particles, nsol)
        return sol

    def reshape_obs(self, obs):
        # obs : (nx)
        obs = obs.reshape((1, -1))
        # obs : (1, nx)
        obs = obs.expand(self.pop_size * self.n_particles, -1)
        # obs : (pop_size * n_particles, nx)
        return obs




