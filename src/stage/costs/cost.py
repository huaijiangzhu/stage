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

    def forward(self, x, a, t, terminal):
        raise NotImplementedError

    def lx(self, l, x, t):
        raise NotImplementedError

    def lxx(self, lx, x, t):
        raise NotImplementedError

    def la(self, l, a, t):
        raise NotImplementedError

    def lax(self, la, x, t):
        raise NotImplementedError

    def laa(self, la, a, t):
        raise NotImplementedError


