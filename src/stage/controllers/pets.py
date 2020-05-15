import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from stage.controllers.base import Controller
from stage.optimizers.cem import CEM
from stage.utils.nn import truncated_normal

class PETS(nn.Module):

    def __init__(self, dynamics, cost, actor,
                 plan_horizon, n_particles, pop_size):
        super().__init__()
        self.nx, self.nq, self.nv = dynamics.nx, dynamics.nq, dynamics.nv
        self.actor = actor
        self.na = actor.na
        self.dynamics = dynamics
        self.action_ub, self.action_lb = actor.action_ub, actor.action_lb

        self.plan_horizon, self.n_particles = plan_horizon, n_particles
        self.pop_size = pop_size

        self.optimizer = CEM(nsol=self.na*self.plan_horizon, pop_size=self.pop_size,
                             ub=self.action_ub.repeat(self.plan_horizon),
                             lb=self.action_lb.repeat(self.plan_horizon))
        self.cost = PETSCost(self.plan_horizon, self.n_particles, self.pop_size,
                              self.dynamics, self.actor, cost)
        self.init_var = ((self.action_ub - self.action_lb) **2 / 16).repeat(self.plan_horizon)
        self.reset()

    def regularize(self, ns):
        self.cost.ns = ns

    def reset(self):
        self.optimizer.reset(nsol=self.plan_horizon*self.na,
                             ub=self.action_ub.repeat(self.plan_horizon),
                             lb=self.action_lb.repeat(self.plan_horizon))
        self.optimizer.horizon = self.plan_horizon
        self.prev_sol = ((self.action_lb + self.action_ub)/2).repeat(self.plan_horizon)

    @torch.no_grad()
    def openloop(self, x, init_sol, horizon):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

        self.cost.obs = x
        self.cost.horizon = horizon

        init_var = ((self.action_ub - self.action_lb) **2 / 16).repeat(horizon)
        self.optimizer.reset(nsol=horizon*self.na,
                             ub=self.action_ub.repeat(horizon),
                             lb=self.action_lb.repeat(horizon))
        self.optimizer.horizon = horizon
        sol, _, opt = self.optimizer(self.cost, init_sol, init_var)
        return sol

    @torch.no_grad()
    def forward(self, x, params=None, random=False):
        if random:
            a = self.actor.sample()

        else:
            sol = self.openloop(x, self.prev_sol, self.plan_horizon)
            self.prev_sol = torch.cat((sol[self.na:], (self.action_lb + self.action_ub)/2))
            a = sol[:self.na]

        return a

class PETSCost(nn.Module):
    def __init__(self, horizon, n_particles, pop_size,
                 dynamics, actor, cost):
        super().__init__()
        self.dynamics = dynamics
        self.actor = actor
        self.nq, self.nv = dynamics.nq, dynamics.nv
        self.na = actor.na
        self.ensemble_size = dynamics.ensemble_size
        self.horizon = horizon
        self.n_particles = n_particles
        self.pop_size = pop_size
        self.cost = cost

        # number of samples for Lipschitz regularization
        self.ns = 0

    def forward(self, action_traj):
        action_traj = self.reshape_action_traj(action_traj)
        obs = self.reshape_obs(self.obs)
        costs = torch.zeros(self.pop_size, self.horizon)  
        ns = self.ns      

        for n in range(self.horizon):
            obs.requires_grad_(True)
            a = action_traj[n]
            b, _ = obs.shape
            obs = obs.repeat(ns + 1, 1)
            a = a.repeat(ns + 1, 1)

            

            # regularize Lipschitz samples
            perturbation = torch.empty(obs.shape).normal_(mean=0, std=0.1)
            perturbation[:b] = 0
            prediction = self.dynamics.sample_predictions(obs + perturbation, a, self.n_particles)
            next_obs = prediction.x

            if ns > 0:
                ref = next_obs[:b]
                ref = ref.repeat(ns + 1, 1)
                deviation = next_obs - ref
                norm_deviation = torch.norm(deviation, p=2, dim=1)
                norm_pertubation = torch.norm(perturbation, p=2, dim=1)
                L = norm_deviation[b:]/norm_pertubation[b:]
                reg = 0
                for s in range(ns):
                    reg += L[b*s:b*(s+1)]
                reg = reg/ns
                cost = self.cost.l(next_obs[:b], a[:b]).l + 0.1 * reg.view(-1, 1)
            else:
                cost = self.cost.l(next_obs[:b], a[:b]).l

            cost = cost.view(-1, self.n_particles)
            costs[:, n] = cost.mean(dim=1)
            obs = next_obs[:b]

        return costs

    def reshape_action_traj(self, action_traj):
        # action_traj : (pop_size, horizon*nu)
        action_traj = action_traj.view(-1, self.horizon, self.na)
        # action_traj : (pop_size, horizon, nu)
        transposed = action_traj.transpose(0, 1)
        # action_traj : (horizon, pop_size, nu)
        expanded = transposed[:, :, None]
        # action_traj : (horizon, pop_size, 1, nu)
        tiled = expanded.expand(-1, -1, self.n_particles, -1)
        # action_traj : (horizon, pop_size, n_particles, nu)
        action_traj = tiled.contiguous().view(self.horizon, -1, self.na)
        # action_traj : (horizon, pop_size * n_particles, nu)
        return action_traj

    def reshape_obs(self, obs):
        # obs : (nx)
        obs = obs.reshape((1, -1))
        # obs : (1, nx)
        obs = obs.expand(self.pop_size * self.n_particles, -1)
        # obs : (pop_size * n_particles, nx)
        return obs



