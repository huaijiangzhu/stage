import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

import gym
import stage.envs
from tqdm import trange

class Learner(object):
    def __init__(self, task):
        self.data_train = None
        self.task = task

    def reset(self):
        raise NotImplementedError
    
    def learn(self):
        raise NotImplementedError

    def save_training_data(self, path):
        assert self.data_train is not None
        np.save(path, self.data_train.detach().cpu().numpy())

