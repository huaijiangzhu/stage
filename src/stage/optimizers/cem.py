from stage.optimizers.base import Optimizer
from stage.utils.nn import truncated_normal
from scipy.stats import truncnorm
import numpy as np

import torch

class CEM(Optimizer):

    def __init__(self, sol_dim, max_iters, popsize, num_elites,
                 upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        super().__init__()
        self.max_iters, self.popsize, self.num_elites = max_iters, popsize, num_elites
        self.epsilon, self.alpha = epsilon, alpha
        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")
        self.reset(sol_dim, upper_bound, lower_bound)

    def reset(self, sol_dim, upper_bound, lower_bound):
        self.sol_dim = sol_dim
        self.ub, self.lb = upper_bound, lower_bound

    def forward(self, cost_function, init_mean, init_var):

        mean, var, t = init_mean, init_var, 0
        while (t < self.max_iters) and torch.max(var) > self.epsilon:
            
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            samples = truncated_normal((self.popsize, self.sol_dim), mean, torch.sqrt(constrained_var))
            costs = cost_function(samples)

            elites = samples[torch.argsort(costs)][:self.num_elites]
            opt = costs[torch.argsort(costs)][:self.num_elites]
            opt = torch.mean(opt, dim=0)

            new_mean = torch.mean(elites, dim=0)
            new_var = torch.var(elites, dim=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        return mean, var, opt

