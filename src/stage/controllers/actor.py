import torch
import torch.nn as nn

class Actor(nn.Module):

    def __init__(self, na, decoder,
                 action_lb=None, action_ub=None, 
                 normalize=False):
        super().__init__()
        self.na = na
        self.nq, self.nv, self.nu = decoder.nq, decoder.nv, decoder.nu
        self.decoder = decoder
        self.action_lb, self.action_ub = action_lb, action_ub

    def forward(self, x, a):
        return self.decoder(x, a)

    ## TODO: normalize/sampling etc...
        
