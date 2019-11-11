import torch
import torch.nn as nn
from stage.controllers.base import Controller

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
            b, _ = obs.shape
            ns = 1

            u = self.inner_loop_controller(obs, a)
            obs = obs.repeat(ns + 1, 1)
            a = a.repeat(ns + 1, 1)

            # regularize Lipschitz samples
            perturbation = torch.empty(obs.shape).normal_(mean=0, std=0.05)
            perturbation[:b] = 0
            next_obs, mean = self.dynamics.sample_predictions(obs + perturbation, a, self.n_particles)

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
                cost = self.obs_cost(next_obs[:b]) + self.action_cost(u) + 0.1 * reg
            else:
                cost = self.obs_cost(next_obs[:b]) + self.action_cost(u)
            
            cost = cost.view(-1, self.n_particles)
            costs += cost
            obs = next_obs[:b]

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



