import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stage.controllers.base import Controller

class PD(Controller):
    def __init__(self, nq, nv, nu):
        super().__init__(nq, nv, nu)
        self.nx = nq + nv  

    def forward(self, x, params):
        x_dim = x.ndimension()
        params_dim = params.ndimension()
        if x_dim == 1:
            # add batch size dimension
            x = x.unsqueeze(0)
        if params_dim == 1:
            # add batch size dimension
            params = params.unsqueeze(0)

        Kp = params[:, :self.nq]
        Kd = 2 * torch.sqrt(Kp)
        g = params[:, self.nq:2*self.nq]

        q = x[:, :self.nq]
        v = x[:, self.nq:self.nx]
        e = self.wrap(g - q)
        
        return Kp * e - Kd * v


class PDFull(Controller):
    def __init__(self, nq, nv, nu):
        super().__init__(nq, nv, nu)
        self.nx = nq + nv  
    
    def forward(self, x, params):
        x_dim = x.ndimension()
        params_dim = params.ndimension()
        if x_dim == 1:
            # add batch size dimension
            x = x.unsqueeze(0)
        if params_dim == 1:
            # add batch size dimension
            params = params.unsqueeze(0)

        Kp = params[:, :self.nq]
        Kd = params[:, self.nq:2*self.nq] * torch.sqrt(Kp)
        g = params[:, 2*self.nq:3*self.nq]

        q = x[:, :self.nq]
        v = x[:, self.nq:self.nx]
        e = self.wrap(g - q)
        
        return Kp * e - Kd * v


    
