import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stage.controllers.base import Controller
from stage.utils.nn import swish, bmv

class MLP(Controller):
    def __init__(self, nx, nq, nv, nu, shape=[10, 10]):
        super().__init__(nx, nq, nv, nu)
        self.shape = shape

        self.nparams = 0
        nin = self.nx
        for nout in self.shape:
            self.nparams += nout * nin + nout
            nin = nout
        nout = self.nu
        self.nparams += nout * nin + nout

    def forward(self, x, params):
        x_dim = x.ndimension()
        params_dim = params.ndimension()
        if x_dim == 1:
            # add batch size dimension
            x = x.unsqueeze(0)
        if params_dim == 1:
            # add batch size dimension
            params = params.unsqueeze(0)

        nb, dim = x.shape
        W = []
        b = []

        start = 0
        nin = self.nx
        inputs = x

        for nout in self.shape:
            w = params[:, start : start + nout * nin]
            w = w.view(nb, nout, nin)
            b = params[:, start + nout * nin: start + nout * (nin + 1)]
            start = start + nout * (nin + 1)
            nin = nout
            inputs = swish(bmv(w, inputs) + b)

        nout = self.nu
        w = params[:, start : start + nout * nin]
        w = w.view(nb, nout, nin)
        b = params[:, start + nout * nin: start + nout * (nin + 1)]

        return bmv(w, inputs) + b


    
