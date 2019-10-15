import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from stage.controllers.base import Controller
from stage.optimizers.cem import CEM
from stage.utils.nn import truncated_normal

class TSMPCCost(nn.Module):
    def __init__(self, horizon, n_particles, pop_size,
                 dynamics, inner_loop_controller, na, step_cost):
        super().__init__()
        self.dynamics = dynamics
        self.inner_loop_controller = inner_loop_controller
        self.nq, self.nv = dynamics.nq, dynamics.nv
        self.na = na
        self.ensemble_size = dynamics.ensemble_size
        self.horizon = horizon
        self.n_particles = n_particles
        self.pop_size = pop_size
        self.step_cost = step_cost

    def forward(self, action_traj):
        action_traj = self.reshape_action_traj(action_traj)
        obs = self.reshape_obs(self.obs)
        costs = torch.zeros(self.pop_size, self.n_particles)        

        for n in range(self.horizon):
            obs.requires_grad_(True)
            a = action_traj[n]
            next_obs, u = self.dynamics.sample_predictions(obs, a, self.inner_loop_controller, self.n_particles)
            cost = self.obs_cost(next_obs) + self.action_cost(u) 
            cost = cost.view(-1, self.n_particles)
            costs += cost
            obs = next_obs
        
        # Replace nan with high cost
        costs[costs != costs] = 1e6

        return costs.mean(dim=1)

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

    def obs_cost(self, obs):
        return self.step_cost.obs_cost(obs)

    def action_cost(self, action):
        return self.step_cost.action_cost(action)



