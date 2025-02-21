from stage.optimizers.base import Optimizer
from stage.utils.nn import truncated_normal
from scipy.stats import truncnorm
import numpy as np

import torch
import pdb

class CEM(Optimizer):

    def __init__(self, nsol,
                 ub=None, lb=None, 
                 pop_size=400, num_elites=40, max_it=5,
                 epsilon=0.001, alpha=0.1):
        super().__init__()
        self.pop_size, self.max_it = pop_size, max_it
        self.num_elites = int(self.pop_size/10)
        self.epsilon, self.alpha = epsilon, alpha
        self.reset(nsol, ub, lb)

    def reset(self, nsol, ub, lb):
        self.nsol = nsol
        self.ub, self.lb = ub, lb

    def forward(self, cost_function, mean_init, var_init):

        mean, var, i = mean_init, var_init, 0
        while (i < self.max_it) and torch.max(var) > self.epsilon:
            
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            samples = truncated_normal((self.pop_size, self.nsol), mean, torch.sqrt(constrained_var))

            nxa = int(self.nsol / self.horizon)
            # some naive smoothing
            for t in range(1, self.horizon):
                samples[:, nxa*t:nxa*(t+1)] = 0.8*samples[:, nxa*t:nxa*(t+1)] + 0.2*samples[:, nxa*(t-1):nxa*t]

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

        

