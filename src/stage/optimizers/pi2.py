from stage.optimizers.base import Optimizer
from stage.utils.nn import truncated_normal
from scipy.stats import truncnorm
import numpy as np

import torch
import torch.nn.functional as F
import pdb

class PI2(Optimizer):

    def __init__(self, na, horizon,
                 upper_bound=None, lower_bound=None,
                 pop_size=400, max_iters=5, 
                 epsilon=0.001, alpha=0.2, gamma=0.5):
        super().__init__()
        self.na, self.horizon = na, horizon
        self.pop_size, self.max_iters = pop_size, max_iters
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma
        self.reset(na * horizon, upper_bound, lower_bound)

    def reset(self, sol_dim, upper_bound, lower_bound):
        self.sol_dim = sol_dim
        self.ub, self.lb = upper_bound, lower_bound

    def forward(self, cost_function, init_mean, init_var):

        mean, var, t = init_mean, init_var, 0

        while t < self.max_iters:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            samples = truncated_normal((self.pop_size, self.sol_dim), mean, torch.sqrt(constrained_var))

            # smooth the action sequence
            samples_reshaped = samples.view(self.pop_size, self.horizon, self.na)
            for i in range(1, samples_reshaped.shape[1]):
                samples_reshaped[:, i, :] = 0.9 * samples_reshaped[:, i, :] + \
                                            0.1 * samples_reshaped[:, i-1, :]
            samples = samples_reshaped.view(self.pop_size, -1)
            costs = cost_function(samples)

            # Replace NaNs and Infs with the non-inf maximum 
            # Warning: this assumes that there is no -inf in the costs!
            costs_finite = costs[torch.isfinite(costs)]
            if len(costs_finite) != 0:
                max_cost_finite = torch.max(costs_finite)
            else:
                max_cost_finite = 0
            costs[costs != costs] = max_cost_finite
            costs[torch.isinf(costs)] = max_cost_finite

            # Normalize to [0, 10] so that exp behaves well (is this numerically stable?)
            costs = (costs - torch.min(costs))/(torch.max(costs) - torch.min(costs) + 1e-6)
            costs = 10 * costs

            # Weighting the action sequences by softmax
            P = F.softmax(-self.gamma * (costs), dim=0)
            if len(P[P!=P])!=0:
                pdb.set_trace()
            new_mean = torch.sum(P * samples.T, dim=1)
            samples_centered = samples - new_mean
            new_var = torch.diag(torch.mm(P * samples_centered.T, samples_centered))

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            t += 1


        # print (mean[:7])

        return mean, None, None

