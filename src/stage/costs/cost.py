import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from stage.controllers.base import Controller
from stage.optimizers.cem import CEM
from stage.utils.nn import truncated_normal

class Cost(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, u, t, terminal):
        raise NotImplementedError

    def lx(self, l, x, t):
        raise NotImplementedError

    def lxx(self, lx, x, t):
        raise NotImplementedError

    def lu(self, l, u, t):
        raise NotImplementedError

    def lux(self, lu, x, t):
        raise NotImplementedError

    def luu(self, lu, u, t):
        raise NotImplementedError


