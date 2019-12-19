import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from dotmap import DotMap

from stage.controllers.base import Controller

class ILQR(nn.Module):

    def __init__(self, dynamics, cost, actor, horizon,
                 alpha=1.0, decay=0.05):
        super().__init__()
        self.dynamics = dynamics
        self.nx = dynamics.nx
        self.actor = actor
        self.na = actor.na
        self.action_ub, self.action_lb = actor.action_ub, actor.action_lb
        self.cost = cost
        self.cost.actor = actor
        self.horizon = horizon

    def forward_pass(self, x0, action_sequence, horizon):
        traj = DotMap()

        if not isinstance(x0, torch.Tensor):
            x0 = torch.Tensor(x0)

        if action_sequence is None:
            action_sequence = self.actor.sample(horizon)

        X = torch.zeros(horizon, self.nx)

        Fx = torch.zeros(horizon - 1, self.nx, self.nx)
        Fa = torch.zeros(horizon - 1, self.nx, self.na)

        L = torch.zeros(horizon, 1)
        Lx = torch.zeros(horizon, self.nx)
        Lxx = torch.zeros(horizon, self.nx, self.nx)
        
        La = torch.zeros(horizon - 1, self.na)
        Laa = torch.zeros(horizon - 1, self.na, self.na)
        Lax = torch.zeros(horizon - 1, self.na, self.nx)

        X[0] = x0
        for t in range(horizon - 1):
            x = X[t]
            a = action_sequence[t]

            cost = self.cost.l(x.clone().detach(), a.clone().detach(), t, diff=True)
            prediction = self.dynamics.sample_predictions(x.clone().detach(), a.clone().detach(), n_particles=0, diff=True)

            X[t+1] = prediction.x
            Fx[t] = prediction.fx
            Fa[t] = prediction.fa

            L[t] = cost.l
            Lx[t] = cost.lx
            Lxx[t] = cost.lxx

            La[t] = cost.la
            Laa[t] = cost.laa
            Lax[t] = cost.lax

        
        x = X[-1]
        cost = self.cost.l(x.clone().detach(), a.clone().detach(), horizon, terminal=True, diff=True)
        L[-1] = cost.l
        Lx[-1] = cost.lx
        Lxx[-1] = cost.lxx

        traj.X = X
        traj.Fx = Fx
        traj.Fa = Fa
        traj.L = L
        traj.Lx = Lx
        traj.Lxx = Lxx
        traj.La = La
        traj.Laa = Laa
        traj.Lax = Lax

        return traj

    def backward_pass(self, traj):
        pass

    def regularize(self, ns):
        pass

    def reset(self):
        pass




