import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time

import gym
import stage.envs
from tqdm import trange

class Task(object):
    def __init__(self, dt_env, dt_control, step_cost, render):
        self.data_train = None
        self.dt_env = dt_env
        self.dt_control = dt_control
        self.env = gym.make(self.env_name, dt=dt_env, step_cost=step_cost, do_render=render)

    def reset(self):
        self.env.reset()
    
    def learn(self):
        raise NotImplementedError

    def unroll(self, x, controller=None, params_generator=None, random=False):

        # decompose the time-dependent part and state-dependent part of the controller
        # -> params_generator is a function that depends only on time
        # -> controller depends on the state and the params = params_generator(n)
        
        if controller is None:
            controller = self.controller
        if params_generator is None:
            params_generator = lambda a : None

        data = []
        log = []

        for n in range(self.task_horizon):
            params = params_generator(n)
            x, reward, done, info = self.act(x, controller, params, random)
            transition = info.transition
            log.append(info)
            data.append(transition)
            if done:
                break
        data = torch.stack(data)
        return data, log

    def act(self, x, controller, params, random):
        control_repetition = int(self.dt_control/self.dt_env)
        a = controller(x, params, random)
        x0 = x.clone()

        for i in range(control_repetition):
            u = torch.flatten(controller.inner_loop_controller(x, a))
            obs, reward, done, info = self.env.step(u)
            x = torch.Tensor(obs[:self.nx])

        dx = x - x0
        transition = torch.cat((x0, a, dx), dim=0)
        info.transition = transition
        return x, reward, done, info


    def visualize_training_data(self, data_train=None, it_begin=0):
        ## This only works for episodic tasks
        assert self.env.do_render == True

        if data_train is None:
            assert self.data_train is not None
            data_train = self.data_train

        self.env.reset()
        n_steps = data_train.shape[0]
        n_rollouts = int(n_steps/self.task_horizon)

        for i in range(it_begin, n_rollouts):
            q = data_train[i*self.task_horizon:(i+1)*self.task_horizon, :self.nq]
            v = data_train[i*self.task_horizon:(i+1)*self.task_horizon, self.nq:self.nx]
            time.sleep(2)
            for n in range(self.task_horizon):
                self.env.set_state(q[n], v[n])
                time.sleep(self.dt_control)
        # self.env.close()

    def save_training_data(self, path):
        assert self.data_train is not None
        np.save(path, self.data_train.detach().cpu().numpy())

