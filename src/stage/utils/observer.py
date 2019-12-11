import torch
import torch.nn as nn


class Observer(nn.Module):

    def __init__(self, nobs, decoder,
                 normalize=False):
        super().__init__()
        self.nobs = nobs
        self.decoder = decoder

    def forward(self, obs):
        return self.decoder(obs)

    ## TODO: everything else
        
