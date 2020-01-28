import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

from stage.learners.base import Learner
from stage.controllers.trivial import RandomController
from tqdm import trange

class LearnAndControlModel(Learner):
    epochs = 10
    batch_size = 64
    learning_rate = 0.001

    def __init__(self, task, dynamics, controller):
        super().__init__(task)
        self.dynamics = dynamics
        self.controller = controller
    
    def learn(self, iteration, verbose=False):
        logs = []

        for i in range(iteration):
        
            x = self.task.reset()
            self.controller.restart()
            start = time.time()
            
            if self.data_train is None:
                random_controller = RandomController(self.task.nx, self.controller.actor)
                self.data_train, log = self.task.unroll(x, random_controller, random=True)
                end = time.time()
                self.dynamics.learn(self.data_train, self.epochs, batch_size=self.batch_size, verbose=verbose)
            else:
                self.controller.update(x)
                new_data, log = self.task.unroll(x, self.controller, random=False)
                end = time.time()
                self.data_train = torch.cat((self.data_train, new_data), dim=0)
                self.dynamics.learn(self.data_train, self.epochs, verbose=verbose)
            
            if verbose:
                print ('Iteration: ', i)
                ro = np.sum([log[i].obs_reward for i in range(self.task.task_horizon)])
                ra = np.sum([log[i].act_reward for i in range(self.task.task_horizon)])
                print ("avg. decision time: ", (end - start)/self.task.task_horizon)
                print ("obs. reward: ", ro)
                print ("act. reward: ", ra)
                
            logs.append(log)
        return logs
    