import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Controller(nn.Module):
    def __init__(self, nx, nq, nv, nu):
        super().__init__()
        self.nx, self.nq, self.nv, self.nu = nx, nq, nv, nu
        
    def forward(self, x, params):
        raise NotImplementedError

    def wrap(self, q):
        return torch.atan2(torch.sin(q), torch.cos(q))

    def get_dim(self):
        return self.nx, self.nq, self.nv, self.nu, self.nparams

    def reset(self):
        pass
