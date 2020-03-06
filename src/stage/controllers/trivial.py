import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stage.controllers.base import Controller
from stage.utils.nn import truncated_normal


class Identity(Controller):
    def __init__(self, nx, nq, nv, nu):
        super().__init__(nx, nq, nv, nu)
        self.nparams = self.nu
    
    def forward(self, x, params, random=False):
        return torch.Tensor(params)

class RandomController(Controller):
    def __init__(self, actor):
        super().__init__(actor.nx, actor.nq, actor.nv, actor.nu)
        self.nx, self.na = actor.nx, actor.na
        self.actor = actor
        self.action_lb = actor.action_lb
        self.action_ub = actor.action_ub
    
    @torch.no_grad()
    def forward(self, x, params, random=True):
        mean = (self.action_lb + self.action_ub)/2
        var =  ((self.action_ub - self.action_lb)**2)/16
        lb_dist, ub_dist = mean - self.action_lb, self.action_ub - mean
        constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
        a = truncated_normal(mean.shape, mean, torch.sqrt(constrained_var))
        return a

class OpenLoop(Controller):
    def __init__(self, actor, actions):
        super().__init__(actor.nx, actor.nq, actor.nv, actor.nu)
        self.nx, self.na = actor.nx, actor.na
        
        ## TODO some dim. check here
        self.actions = torch.Tensor(actions)
        self.actor = actor
    
    @torch.no_grad()
    def forward(self, x, params, random=False):
        t = params
        a = self.actions[t, :]
        return a

    
