import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange
from dotmap import DotMap

from stage.controllers.base import Controller
from stage.utils.nn import truncated_normal

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
        self.horizon = horizon


    @torch.no_grad()
    def forward_pass(self, x, action_sequence, horizon):
        traj = DotMap()

        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)

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

        pass

    @torch.no_grad()
    def backward_pass(self, traj):
        pass

    def regularize(self, ns):
        pass

    def reset(self):
        pass




