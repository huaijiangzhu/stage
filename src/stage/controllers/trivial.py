import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stage.controllers.base import Controller

class Identity(Controller):
    def __init__(self, nq, nv, nu):
        super().__init__(nq, nv, nu)
        self.nx = nq + nv  
    
    @torch.no_grad()
    def forward(self, x, params, random=False):
        return torch.Tensor(params)

class OpenLoop(Controller):
    def __init__(self, nq, nv, nu, na, action_sequence, actor):
        super().__init__(nq, nv, nu)
        self.nx = nq + nv
        self.na = na
        ## TODO some dim. check here
        self.action_sequence = torch.Tensor(action_sequence)
        self.actor = actor
    
    @torch.no_grad()
    def forward(self, x, params, random=False):
        t = params
        a = self.action_sequence[t, :]
        return a

    
