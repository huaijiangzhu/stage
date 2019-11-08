from stage.optimizers.base import Optimizer
from stage.utils.nn import truncated_normal
from scipy.stats import truncnorm
import numpy as np

import torch

class PI2(Optimizer):

    def __init__(self, na, horizon, popsize
                 upper_bound=None, lower_bound=None, gamma=0.1):
        super().__init__()
        self.na = na
        self.horizon = horizon
        self.popsize = popsize
        self.gamma = gamma
        self.reset(self.na * self.horizon, upper_bound, lower_bound)

    def reset(self, sol_dim, upper_bound, lower_bound):
        self.sol_dim = sol_dim
        self.ub, self.lb = upper_bound, lower_bound

    def forward(self, cost_function, init_mean, init_var):

        mean, var, t = init_mean, init_var, 0
        while (t < self.max_iters):
            
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            samples = truncated_normal((self.popsize, self.sol_dim), mean, torch.sqrt(constrained_var))

            # smooth the action sequence
            samples_reshaped = samples.view(self.popsize, self.horizon, self.na)
            for i in range(1, samples_reshaped.shape[1]):
                samples_reshaped[:, i, :] = 0.9 * samples_reshaped[:, i, :] + \
                                            0.1 * samples_reshaped[:, i-1, :]
            samples = samples_reshaped.view(self.popsize, -1)
            costs = cost_function(samples)

            # averaging the sequences by cost exponentionals
            costs -= torch.mean(costs)
            S = torch.exp(-0.01*(costs))
            S = S.reshape((-1, 1))
            S = S.expand(-1, samples.shape[1])
            samples_weighted = S * samples
            new_mean = torch.sum(samples_weighted, dim=0)/torch.sum(S)

            # new_var = torch.var(elites, dim=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            # var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1

        return mean, None, None

