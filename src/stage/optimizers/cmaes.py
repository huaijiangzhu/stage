from stage.optimizers.base import Optimizer
from stage.utils.nn import truncated_normal
from scipy.stats import truncnorm
import numpy as np

import torch
import torch.nn.functional as F
import pdb

class CMAES(Optimizer):

    def __init__(self, na, horizon,
                 upper_bound=None, lower_bound=None,
                 pop_size=400, max_iters=5, 
                 epsilon=0.001, alpha=0.1, gamma=0.75):
        super().__init__()
        self.na, self.horizon = na, horizon
        self.pop_size, self.max_iters = pop_size, max_iters
        self.epsilon, self.alpha, self.gamma = epsilon, alpha, gamma
        self.reset(na * horizon, upper_bound, lower_bound)
        self.num_elites = int(self.pop_size/5)

    def reset(self, sol_dim, upper_bound, lower_bound):
        self.sol_dim = sol_dim
        self.ub, self.lb = upper_bound, lower_bound

    def forward(self, cost_function, init_mean, init_var):

        mean, var, i = init_mean, init_var, 0

        while i < self.max_iters:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = torch.min(torch.min((lb_dist / 2)**2, (ub_dist / 2)**2), var)
            samples = truncated_normal((self.pop_size, self.sol_dim), mean, torch.sqrt(constrained_var))
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

            # Pick elites
            # Theoretically we should be able to have num_elites = pop_size
            # But that leads to slow convergence in the beginning of the training
            costs, idx_costs = torch.sort(costs)
            elites = samples[idx_costs][:self.num_elites]
            costs = costs[:self.num_elites]

            # Normalize to [0, 10] so that exp behaves well (is this numerically stable?)
            costs = (costs - torch.min(costs))/(torch.max(costs) - torch.min(costs) + 1e-6)
            costs = 10 * costs

            # Weighting the action sequences by softmax
            P = F.softmax(-self.gamma * (costs), dim=0)
            new_mean = torch.sum(P * elites.T, dim=1)
            elites_centered = elites - mean
            new_var = torch.diag(torch.mm(P * elites_centered.T, elites_centered))

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var
            i += 1

        return mean, var, None

