import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dotmap import DotMap
import numpy as np

from stage.dynamics.base import Dynamics
from stage.utils.nn import swish, get_affine_params, truncated_normal
from stage.utils.jacobian import JacobianNormEnsemble, AutoDiffEnsemble
from stage.utils.nn import renew
import tqdm

import pdb

class ProbabilisticEnsemble(Dynamics):
    def __init__(self, nx, nq, nv, na, dt, dx, 
                 ensemble_size=5,
                 learning_rate=0.005):
        super().__init__(nx, nq, nv, na, dt)
        self.nin = self.nx + self.na
        self.nout = 2 * self.nx
        self.dx = dx(ensemble_size, self.nx, self.nq, self.na)
        self.opt = optim.Adam(self.dx.parameters(), lr=learning_rate)
        self.ensemble_size = ensemble_size
        self.d = AutoDiffEnsemble()

    def fx(self, f, x):
        return self.d(f, x)

    def fa(self, f, a):
        return self.d(f, a)

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
        # TODO: support unroll with different initial conditions/multiple action sequences
        
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
            prediction = self.sample_predictions(obs, a, n_particles, diff=True)
            obs = prediction.x
        obs_traj = torch.stack(obs_traj, dim=0)
        return obs_traj

    def sample_predictions(self, x, a, n_particles, diff=False):

        # n_particles = 0 --> no sampling
        if n_particles > 0:
            assert n_particles % self.ensemble_size == 0
        
        x_dim, a_dim = x.ndimension(), a.ndimension()
        if x_dim == 1:
            x = x.unsqueeze(0)
        if a_dim == 1:
            a = a.unsqueeze(0)

        if n_particles > 0:
            x = self.group_as_ensemble(x, n_particles)
            a = self.group_as_ensemble(a, n_particles)
        else:
            x = torch.stack(self.ensemble_size * [x])
            a = torch.stack(self.ensemble_size * [a])
        if diff:
            x, a = renew(x), renew(a)
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

        if n_particles > 0:
            prediction.x = self.flatten_ensemble(x_, n_particles)
        else:
            prediction.x = torch.mean(x_, dim=0)
        
        return prediction


    def group_as_ensemble(self, arr, n_particles):
        dim = arr.shape[-1]
        # mat : [b, dim] 
        reshaped = arr.view(-1, self.ensemble_size, n_particles // self.ensemble_size, dim)
        # mat : [b // n_particles, ensemble_size, n_particles // ensemble_size, dim]
        transposed = reshaped.transpose(0, 1)
        # mat : [ensemble_size, b // n_particles, n_particles // ensemble_size, dim]
        reshaped = transposed.contiguous().view(self.ensemble_size, -1, dim)
        # mat : [ensemble_size, b // ensemble_size, dim]
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
        self.dx.train()
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
        self.dx.eval()

class Dx(nn.Module):
    def __init__(self, ensemble_size, nx, nq, na, first_joint_id=0):
        super().__init__()
        self.ensemble_size, self.nx, self.nq, self.na = ensemble_size, nx, nq, na
        self.first_joint_id = 0
        self.nin = nx + na
        self.nout = 2 * nx
        self.data_mu = nn.Parameter(torch.zeros(self.nin + self.nq), requires_grad=False)
        self.data_sigma = nn.Parameter(torch.zeros(self.nin + self.nq), requires_grad=False)
        self.max_logvar = nn.Parameter(torch.ones(1, self.nx) / 2.0)
        self.min_logvar = nn.Parameter(-torch.ones(1, self.nx) * 10.0)
        self.jac_norm = JacobianNormEnsemble()
        self.lambda_jac_reg = 0.01

    def normalize(self, inputs):
        nb, dim = inputs.shape
        inputs = self.embed(inputs.view(1, nb, dim)).view(nb, dim + self.nq)
        mu = inputs.mean(dim=0)
        sigma = inputs.std(dim=0)
        sigma[sigma < 1e-12] = 1.0
        self.data_mu.data = mu.data
        self.data_sigma.data = sigma.data

    def compute_loss(self, xa, dx, mse=False):
        # regularization
        reg = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        reg += self.compute_decays()

        xa.requires_grad = True
        dx_pred, logvar_pred = self.forward(xa, return_logvar=True)
        jac_norm = self.jac_norm(dx_pred, xa)
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

    def embed(self, inputs):
        start = self.first_joint_id
        ensemble_size, nb, dim = inputs.shape

        embedded = torch.zeros(ensemble_size, nb, dim + self.nq)
        q = inputs[:, :, start : start + self.nq]
        q = q.view(ensemble_size * nb, self.nq)
        
        cos = torch.cos(q)
        sin = torch.sin(q)
        cs = torch.cat((cos, sin), dim=-1)
        cs = cs.view(ensemble_size, nb, 2 * self.nq)

        embedded[:, :, :start] = inputs[:, :, :start]
        embedded[:, :, start : start + 2 * self.nq] = cs
        embedded[:, :, start + 2 * self.nq:] = inputs[:, :, start + self.nq:]
        return embedded
        

class DefaultDx(Dx):
    def __init__(self, ensemble_size, nx, nq, na, first_joint_id=0):
        super().__init__(ensemble_size, nx, nq, na, first_joint_id)
        self.lin0_w, self.lin0_b = get_affine_params(self.ensemble_size, self.nin + self.nq, 300)
        self.lin1_w, self.lin1_b = get_affine_params(self.ensemble_size, 300, 300)
        self.lin2_w, self.lin2_b = get_affine_params(self.ensemble_size, 300, 300)
        self.lin3_w, self.lin3_b = get_affine_params(self.ensemble_size, 300, 300)
        self.lin4_w, self.lin4_b = get_affine_params(self.ensemble_size, 300, self.nout)
        
    def forward(self, inputs, return_logvar=False):
        inputs = self.embed(inputs)

        inputs = (inputs - self.data_mu) / self.data_sigma

        inputs = inputs.matmul(self.lin0_w) + self.lin0_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin1_w) + self.lin1_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin2_w) + self.lin2_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin3_w) + self.lin3_b
        inputs = swish(inputs)

        inputs = inputs.matmul(self.lin4_w) + self.lin4_b

        mean = inputs[:, :, :self.nx]
        logvar = inputs[:, :, self.nx:]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)

        if return_logvar:
            return mean, logvar
        else:
            return mean, torch.exp(logvar)

    def compute_decays(self):

        lin0_decays = 0.00025 * (self.lin0_w ** 2).sum() / 2.0
        lin1_decays = 0.0005 * (self.lin1_w ** 2).sum() / 2.0
        lin2_decays = 0.0005 * (self.lin2_w ** 2).sum() / 2.0
        lin3_decays = 0.0005 * (self.lin3_w ** 2).sum() / 2.0
        lin4_decays = 0.00075 * (self.lin4_w ** 2).sum() / 2.0

        return lin0_decays + lin1_decays + lin2_decays + lin3_decays + lin4_decays





