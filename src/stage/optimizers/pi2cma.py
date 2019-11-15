from stage.optimizers.base import Optimizer
from stage.utils.nn import truncated_normal, flip
from scipy.stats import truncnorm
import numpy as np

import torch
import torch.nn.functional as F
import pdb

class PI2CMA(Optimizer):

    def __init__(self, na, horizon,
                 upper_bound=None, lower_bound=None,
                 pop_size=400, max_iters=5, 
                 epsilon=0.001, alpha=0.1, h=10.0):
        super().__init__()
        self.na, self.horizon = na, horizon
        self.pop_size, self.max_iters = pop_size, max_iters
        self.epsilon, self.alpha, self.h = epsilon, alpha, h
        self.reset(na * horizon, upper_bound, lower_bound)

    def reset(self, sol_dim, upper_bound, lower_bound):
        self.sol_dim = sol_dim
        self.ub, self.lb = upper_bound, lower_bound

    def forward(self, cost_function, init_mean, init_var):

        mean, var, i = init_mean, init_var, 0

        while i < self.max_iters and torch.max(var) > self.epsilon:
            self.h = 50.0 / torch.mean(var)

            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            samples = truncated_normal((self.pop_size, self.sol_dim), mean, torch.sqrt(constrained_var))
            costs = cost_function(samples)

            # Compute cost-to-go
            costs_flipped = flip(costs, dim=1)
            S = torch.zeros_like(costs)
            S[:, 0] = costs_flipped[:, 0]
            for k in range(1, self.horizon):
                S[:, k] = costs_flipped[:, k] + S[:, k - 1]
            S = flip(S, dim=1)

            # Replace NaNs and Infs with the non-inf maximum 
            # Warning: this assumes that there is no -inf in the costs!
            costs_finite = S[torch.isfinite(S)]
            if len(costs_finite) != 0:
                max_cost_finite = torch.max(costs_finite)
            else:
                max_cost_finite = 0
            S[S != S] = max_cost_finite
            S[torch.isinf(S)] = max_cost_finite

            # Normalize so that exp behaves well
            Smin = torch.min(S, dim=0)[0]
            Smax = torch.max(S, dim=0)[0]
            S = (S - Smin)/(Smax - Smin + 1e-6)

            # Weighting the action sequences by softmax
            P = F.softmax(-self.h * (S), dim=0)
            P = P.view(self.pop_size, self.horizon, 1)
            samples = samples.view(self.pop_size, self.horizon, self.na)

            samples_weighted = P * samples
            new_mean = samples_weighted.sum(dim=0)
            samples_centered = samples - new_mean
             
            temp = P * samples_centered
            temp = temp.view(self.pop_size, -1)
            samples_centered = samples_centered.view(self.pop_size, -1)

            new_mean = new_mean.view(self.sol_dim)
            new_var = torch.diag(torch.mm(temp.T, samples_centered))
            
            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            i += 1

        return mean, var, None

