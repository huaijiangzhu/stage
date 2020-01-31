import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from dotmap import DotMap
import tqdm

from stage.dynamics.base import Dynamics
from stage.utils.nn import swish
from stage.utils.jacobian import JacobianNorm, AutoDiff
from stage.utils.nn import renew


class MLPDyn(Dynamics):
    def __init__(self, nx, nq, nv, na, dt, dx, learning_rate=0.001):
        super().__init__(nx, nq, nv, na, dt)
        self.nin = self.nx + self.na
        self.nout = 2 * self.nx
        self.dx = dx(self.nx, self.na)
        self.opt = optim.Adam(self.dx.parameters(), lr=learning_rate)
        self.d = AutoDiff()

    def fx(self, f, x):
        return self.d(f, x)

    def fa(self, f, a):
        return self.d(f, a)

    def forward(self, x, a):
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
            prediction = self.sample_predictions(obs, a, n_particles, diff=False)
            obs = prediction.x
        obs_traj = torch.stack(obs_traj, dim=0)
        return obs_traj

    def sample_predictions(self, x, a, n_particles, diff):

        # n_particles <= 0 --> no sampling
        x, a = renew(x), renew(a)        
        x_dim, a_dim = x.ndimension(), a.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if a_dim == 1:
            a = a.unsqueeze(0)
        if diff:
            x.requires_grad = True
            a.requires_grad = True
        mean, var = self.forward(x, a)
        if n_particles > 0:
            x_ = mean + torch.randn_like(mean) * var.sqrt()
        else:
            x_ = mean

        prediction = DotMap()
        if diff:
            prediction.fx = self.d(x_, x)
            prediction.fa = self.d(x_, a)
        prediction.x = x_

        return prediction

    def learn(self, data, epochs, batch_size=32, verbose=False):

        dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
        epoch_range = tqdm.trange(epochs, unit="epoch(s)", desc="Network training", disable=not verbose)
        self.dx.train()
        for _ in epoch_range:
            for _, sample in enumerate(dataloader):

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
        self.dx.eval()

class Dx(nn.Module):
    def __init__(self, nx, na):
        super().__init__()
        self.nx, self.na = nx, na
        self.nin = nx + na
        self.nout = 2 * nx
        self.max_logvar = nn.Parameter(torch.ones(1, self.nx) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, self.nx) * 10.0)
        self.jac_norm = JacobianNorm()
        self.lambda_jac_reg = 0.01

    def compute_loss(self, xa, dx, mse=False):
        # regularization
        
        xa.requires_grad = True
        dx_pred, logvar_pred = self.forward(xa, return_logvar=True)
        jac_norm = self.jac_norm(dx_pred, xa)
        reg = self.lambda_jac_reg * jac_norm
        reg += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        
        inv_var_pred = torch.exp(-logvar_pred)

        if mse:
            # mse loss
            loss = ((dx_pred - dx) ** 2)
        else:
            # nll loss + regularization
            loss = ((dx_pred - dx) ** 2) * inv_var_pred + logvar_pred + reg

        loss = torch.mean(loss)

        return loss

class DefaultDx(Dx):
    def __init__(self, nx, na):
        super().__init__(nx, na)
        self.fc1 = nn.Linear(self.nin, 300)
        self.fc2 = nn.Linear(300, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, self.nout)
        self.bn1 = nn.BatchNorm1d(300)
        self.bn2 = nn.BatchNorm1d(300)
        self.bn3 = nn.BatchNorm1d(300)
        self.dropout = nn.Dropout(0.8)
    
    def forward(self, inputs, return_logvar=False):

        inputs = swish(self.bn1(self.fc1(inputs)))
        inputs = swish(self.bn2(self.fc2(inputs)))
        inputs = self.dropout(inputs)
        inputs = swish(self.bn3(self.fc3(inputs)))
        inputs = self.fc4(inputs)

        mean = inputs[:, :self.nx]
        logvar = inputs[:, self.nx:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if return_logvar:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)







