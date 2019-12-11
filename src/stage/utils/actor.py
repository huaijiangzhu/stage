import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, na, decoder,
                 action_lb=None, action_ub=None, 
                 normalize=False):
        super().__init__()
        self.na = na
        self.decoder = decoder
        self.action_lb, self.action_ub = action_lb, action_ub

    def forward(self, obs, a):
        return self.decoder(obs, a)

    ## TODO: normalize/sampling etc...
        
