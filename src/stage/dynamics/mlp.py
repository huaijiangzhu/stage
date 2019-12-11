import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from stage.dynamics.base import Dynamics
from stage.utils.jacobian_reg import JacobianReg
import tqdm

class MLPDyn(Dynamics):
    def __init__(self, nq, nv, na, dt, dx, learning_rate=0.005):
        super().__init__(nx, nq, nv, na, dt)
        self.nin = self.nx + self.na
        self.nout = 2 * self.nx
        self.dx = dx(self.nx, self.na)
        self.opt = optim.Adam(self.dx.parameters(), lr=learning_rate)

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
        horizon, na = action_traj.shape
        obs = obs.reshape((1, -1))
        action_traj = action_traj.view(1, horizon, na)
        # obs : (1, nx)
        # action_traj : (1, horizon, na)
        if n_particles > 0:
            obs = obs.expand(n_particles, -1)
            # obs : (n_particles, nx)
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
        # n_particles <= 0 --> no sampling
        mean, var = self.forward(obs, a)
        if n_particles > 0:
            prediction = mean + torch.randn_like(mean) * var.sqrt()
        else:
            prediction = mean
        return prediction

    def learn(self, data, epochs, batch_size=32, verbose=False):
        self.dx.normalize(data[:, :self.nin])
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        epoch_range = tqdm.trange(epochs, unit="epoch(s)", desc="Network training", disable=not verbose)

        for _ in epoch_range:
            for i, sample in enumerate(dataloader):

                xa = sample[:, :self.nin]
                dx = sample[:, self.nin:]
                self.opt.zero_grad()
                loss = self.dx.compute_loss(xa, dx)
                loss.backward()
                self.opt.step()

            idx = torch.randperm(data.shape[0])[:5000]
            sample = data[idx]
            xa = sample[:, :self.nin]
            dx = sample[:, self.nin:]
            mse = self.dx.compute_loss(xa, dx, mse=True)
            epoch_range.set_postfix({"Training loss MSE": mse.detach().cpu().numpy()})

class Dx(nn.Module):
    def __init__(self, nx, na):
        super().__init__()
        self.nx, self.na = nx, na
        self.nin = nx + na
        self.nout = 2 * nx
        self.inputs_mu = nn.Parameter(torch.zeros(self.nin), requires_grad=False)
        self.inputs_sigma = nn.Parameter(torch.zeros(self.nin), requires_grad=False)
        self.max_logvar = nn.Parameter(torch.ones(1, self.nx) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, self.nx) * 10.0)
        self.jac_reg = JacobianReg()
        self.lambda_jac_reg = 0.01

    def normalize(self, inputs):
        mu = inputs.mean(dim=0)
        sigma = inputs.std(dim=0)
        sigma[sigma < 1e-12] = 1.0

        self.inputs_mu.data = mu.data
        self.inputs_sigma.data = sigma.data

    def compute_loss(self, xa, dx, mse=False):
        # regularization
        reg = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())

        xa.requires_grad = True
        dx_pred, logvar_pred = self.forward(xa, return_logvar=True)
        jac_norm = self.jac_reg(xa, dx_pred)
        reg += self.lambda_jac_reg * jac_norm
        
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



