import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Optimizer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self):
        raise NotImplementedError
