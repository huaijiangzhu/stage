import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import trange

from stage.controllers.base import Controller
from stage.optimizers.cem import CEM
from stage.utils.nn import truncated_normal

class StepCost(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obs, action):
        return self.obs_cost(obs), self.action_cost(action)

    def obs_cost(self, obs):
        raise NotImplementedError

    def action_cost(self, action):
        raise NotImplementedError



