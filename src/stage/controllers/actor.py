import torch
import torch.nn as nn
from stage.utils.nn import truncated_normal

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

    def sample(self, mean=None, var=None, horizon=1):

        lb = self.action_lb
        ub = self.action_ub

        if mean is None:
            mean = (lb + ub)/2
        if var is None:
            var = (ub - lb) **2 / 16

        lb = lb.repeat(horizon)
        ub = ub.repeat(horizon)
        mean = mean.repeat(horizon)
        var = var.repeat(horizon)

        lb_dist, ub_dist = mean - lb, ub - mean
        constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
        action_sequence = truncated_normal(mean.shape, mean, torch.sqrt(constrained_var))

        return action_sequence

    ## TODO: normalize etc...
        
