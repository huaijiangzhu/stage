import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from stage.dynamics.base import Dynamics
import tqdm

class ProbabilisticEnsemble(Dynamics):
    def __init__(self, nq, nv, na, dt, dx, 
                 ensemble_size=5, learn_closed_loop_dynamics=False):
        super().__init__(nq, nv, na, dt)
        self.learn_closed_loop_dynamics = learn_closed_loop_dynamics
        self.nin = self.nx + self.na
        self.nout = 2 * self.nx
        self.dx = dx(ensemble_size, self.nx, self.na)
        self.opt = optim.Adam(self.dx.parameters(), lr=0.001)
        self.ensemble_size = ensemble_size

    def forward(self, x, a):
        x_dim, a_dim = x.ndimension(), a.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if a_dim == 1:
            a = a.unsqueeze(0)
        xa = torch.cat((x, a), dim=-1)
        dx_pred, var_pred = self.dx(xa)
        return x + dx_pred, var_pred

    def unroll(self, obs, action_traj, n_particles=20):
        # TODO: support unroll with different initial conditions/nultiple action sequences
        assert n_particles % self.ensemble_size == 0
        horizon, na = action_traj.shape
        obs = obs.reshape((1, -1))
        # obs : (1, nx)
        obs = obs.expand(n_particles, -1)
        # obs : (n_particles, nx)
        action_traj = action_traj.view(-1, horizon, na)
        action_traj = action_traj.expand(n_particles, -1, -1)
        # action_traj : (n_particles, horizon, na)
        action_traj = action_traj.transpose(0, 1)
        # action_traj : (horizon, n_particles, na)

        obs_traj = []
        for n in range(horizon):
            obs_traj.append(obs)
            a = action_traj[n]
            next_obs, _ = self.sample_predictions(obs, a, n_particles)
            obs = next_obs
        obs_traj = torch.stack(obs_traj, dim=0)
        return obs_traj

    def sample_predictions(self, obs, a, n_particles):
        obs = self.group_as_ensemble(obs, n_particles)
        mean, var = self.forward(obs, self.group_as_ensemble(a, n_particles))
        samples = mean + torch.randn_like(mean) * var.sqrt()
        samples = self.flatten_ensemble(samples, n_particles)
        mean = self.flatten_ensemble(mean, n_particles)
        return samples, mean

    def group_as_ensemble(self, arr, n_particles):
        dim = arr.shape[-1]
        # mat : [n_particles, dim] 
        reshaped = arr.view(-1, self.ensemble_size, n_particles // self.ensemble_size, dim)
        # mat : [1, ensemble_size, n_particles // ensemble_size, dim]
        transposed = reshaped.transpose(0, 1)
        # mat : [ensemble_size, 1, n_particles // ensemble_size, dim]
        reshaped = transposed.contiguous().view(self.ensemble_size, -1, dim)
        # mat : [ensemble_size, (n_particles) // ensemble_size, dim]
        return reshaped

    def flatten_ensemble(self, arr, n_particles):
        dim = arr.shape[-1]
        reshaped = arr.view(self.ensemble_size, -1, n_particles // self.ensemble_size, dim)
        transposed = reshaped.transpose(0, 1)
        reshaped = transposed.contiguous().view(-1, dim)
        return reshaped

    def learn(self, data, epochs, batch_size=32, verbose=False):
        self.dx.normalize(data[:, :self.nin])
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        epoch_range = tqdm.trange(epochs, unit="epoch(s)", desc="Network training", disable=not verbose)
        for _ in epoch_range:
            for i, sample in enumerate(dataloader):

                xa = sample[:, :self.nin]
                xa = xa.repeat(self.ensemble_size, 1, 1)
                dx = sample[:, self.nin:]
                dx = dx.repeat(self.ensemble_size, 1, 1)
                self.opt.zero_grad()
                loss = self.dx.compute_loss(xa, dx)
                loss.backward()
                self.opt.step()

            idx = torch.randperm(data.shape[0])[:5000]
            sample = data[idx]
            xa = sample[:, :self.nin]
            dx = sample[:, self.nin:]
            xa = xa.repeat(self.ensemble_size, 1, 1)
            dx = dx.repeat(self.ensemble_size, 1, 1)
            mse = self.dx.compute_loss(xa, dx, mse=True)
            epoch_range.set_postfix({"Training loss MSE": mse.detach().cpu().numpy()})

class Dx(nn.Module):
    def __init__(self, ensemble_size, nx, na):
        super().__init__()
        self.ensemble_size, self.nx, self.na = ensemble_size, nx, na
        self.nin = nx + na
        self.nout = 2 * nx
        self.inputs_mu = nn.Parameter(torch.zeros(self.nin), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(self.nin), requires_grad=False)
        self.max_logvar = nn.Parameter(torch.ones(1, self.nx) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, self.nx) * 10.0)

    def normalize(self, inputs):
        mu = inputs.mean(dim=0)
        sigma = inputs.std(dim=0)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = mu.data
        self.inputs_sigma.data = sigma.data

    def compute_loss(self, xa, dx, mse=False):
        # regularization
        reg = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        reg += self.compute_decays()

        dx_pred, logvar_pred = self.forward(xa, return_logvar=True)
        inv_var_pred = torch.exp(-logvar_pred)

        if mse:
            # mse loss
            loss = ((dx_pred - dx) ** 2)
            loss = torch.mean(loss)
        else:
            # nll loss + regularization
            loss = ((dx_pred - dx) ** 2) * inv_var_pred + logvar_pred
            loss = torch.mean(loss)
            loss += reg            

        return loss


