from stage.optimizers.base import Optimizer
from stage.utils.nn import truncated_normal
from scipy.stats import truncnorm
import numpy as np

import torch
import pdb

class CEM(Optimizer):

    def __init__(self, na, horizon,
                 upper_bound=None, lower_bound=None, 
                 pop_size=400, num_elites=40, max_iters=5,
                 epsilon=0.001, alpha=0.1):
        super().__init__()
        self.na, self.horizon = na, horizon
        self.pop_size, self.max_iters = pop_size, max_iters
        self.num_elites = int(self.pop_size/10)
        self.epsilon, self.alpha = epsilon, alpha
        self.reset(na * horizon, upper_bound, lower_bound)

    def reset(self, sol_dim, upper_bound, lower_bound):
        self.sol_dim = sol_dim
        self.ub, self.lb = upper_bound, lower_bound

    def forward(self, cost_function, init_mean, init_var):

        mean, var, i = init_mean, init_var, 0
        while (i < self.max_iters) and torch.max(var) > self.epsilon:
            
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            samples = truncated_normal((self.pop_size, self.sol_dim), mean, torch.sqrt(constrained_var))

            costs = cost_function(samples)
            costs = torch.sum(costs, dim=1)
            
            if len(samples[samples != samples])!=0:
                pdb.set_trace()

            # Replace NaNs and Infs with the non-inf maximum 
            # Warning: this assumes that there is no -inf in the costs!
            costs_finite = costs[torch.isfinite(costs)]
            if len(costs_finite) != 0:
                max_cost_finite = torch.max(costs_finite)
            else:
                max_cost_finite = 0
            costs[costs != costs] = max_cost_finite
            costs[torch.isinf(costs)] = max_cost_finite
            
            elites = samples[torch.argsort(costs)][:self.num_elites]
            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, dim=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            i += 1

        # print (cost_function(mean.expand(self.pop_size, -1))[0])

        return mean, var, None

