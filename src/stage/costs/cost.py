import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from stage.controllers.base import Controller
from stage.optimizers.cem import CEM
from stage.utils.nn import truncated_normal

from stage.utils.nn import bquad, flatten_non_batch
from stage.utils.jacobian import AutoDiff
from stage.utils.nn import renew

class Cost(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, u, t, terminal):
        raise NotImplementedError

    ### Only autodiff is surported 

    def l(self, x, a, t=0, terminal=False, diff=False):
        if diff:
            x, a = renew(x), renew(a)
            x.requires_grad = True
            a.requires_grad = True

        if x.ndimension() == 1:
            x = x.unsqueeze(0)
        if a.ndimension() == 1:
            a = a.unsqueeze(0) 

        u = self.actor(x, a)
        cost = self.forward(x, u, t, terminal)

        if diff:
            cost.lx = flatten_non_batch(self.lx(cost.l, x, a, t))
            cost.lxx = self.lxx(cost.lx, x, a, t)

            if not terminal:
                cost.la = flatten_non_batch(self.la(cost.l, x, a, t))
                cost.lax = self.lax(cost.la, x, a, t)
                cost.laa = self.laa(cost.la, x, a, t)

        return cost

    def lx(self, l, x, a, t):
        return self.d(l, x)

    def lxx(self, lx, x, a, t):
        return self.d(lx, x)

    def la(self, l, x, a, t):
        return self.d(l, a)

    def laa(self, la, x, a, t):
        return self.d(la, a)

    def lax(self, la, x, a, t):
        return self.d(la, x)


