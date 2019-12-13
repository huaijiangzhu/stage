import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Controller(nn.Module):
    def __init__(self, nq, nv, nu):
        super(Controller, self).__init__()
        self.nq = nq
        self.nv = nv
        self.nu = nu
        self.nx = nq + nv

    def forward(self, x, params):
        raise NotImplementedError

    def wrap(self, q):
        return torch.atan2(torch.sin(q), torch.cos(q))

    def reset(self):
        pass
