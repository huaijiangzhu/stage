import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Dynamics(nn.Module):
    def __init__(self, nx, nq, nv, na, dt):
        super().__init__()
        self.nx = nx
        self.nq = nq
        self.nv = nv
        self.na = na
        self.dt = dt

    def forward(self, x, a):
        raise NotImplementedError

    def wrap(self, q):
        return torch.atan2(torch.sin(q), torch.cos(q))

