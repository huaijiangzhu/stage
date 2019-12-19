'''
    Copyright (c) Facebook, Inc. and its affiliates.

    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
    
    PyTorch implementation of Jacobian regularization described in [1].

    [1] Judy Hoffman, Daniel A. Roberts, and Sho Yaida,
        "Robust Learning with Jacobian Regularization," 2019.
        [arxiv:1908.02729](https://arxiv.org/abs/1908.02729)
'''
from __future__ import division
import torch
import torch.nn as nn
from stage.utils.nn import flatten_non_batch, jacobian_vector_product

class JacobianNormEnsemble(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output 
            space and projection is non-random and orthonormal, yielding 
            the exact result.  For any reasonable batch size, the default 
            (n=1) should be sufficient.
    '''
    def __init__(self, n=1):
        assert n == -1 or n > 0
        self.n = n
        super().__init__()

    def forward(self, y, x):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        nensemble, nbatch, ny = y.shape
        if self.n == -1:
            num_proj = ny
        else:
            num_proj = self.n

        J2 = 0
        for i in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v = torch.zeros(nensemble, nbatch, ny)
                v[:, :, i] = 1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(nensemble, nbatch, ny)
            if x.is_cuda:
                v = v.cuda()
            vJ = jacobian_vector_product(y, x, v, create_graph=True)
            J2 += ny * torch.norm(vJ)**2 / (num_proj * nbatch * nensemble)
        R = 1/2 * J2
        return R

    def _random_vector(self, nensemble, nbatch, nv):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if nv == 1: 
            return torch.ones(nensemble, nbatch)
        v = torch.randn(nensemble, nbatch, nv)
        arxilirary_zero=torch.zeros(nensemble, nbatch, nv)
        vnorm = torch.norm(v, 2, -1, True)
        v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v


class JacobianNorm(nn.Module):
    '''
    Loss criterion that computes the trace of the square of the Jacobian.

    Arguments:
        n (int, optional): determines the number of random projections.
            If n=-1, then it is set to the dimension of the output 
            space and projection is non-random and orthonormal, yielding 
            the exact result.  For any reasonable batch size, the default 
            (n=1) should be sufficient.
    '''
    def __init__(self, n=1):
        assert n == -1 or n > 0
        self.n = n
        super().__init__()

    def forward(self, y, x):
        '''
        computes (1/2) tr |dy/dx|^2
        '''
        nbatch, ny = y.shape
        if self.n == -1:
            num_proj = ny
        else:
            num_proj = self.n

        J2 = 0
        for i in range(num_proj):
            if self.n == -1:
                # orthonormal vector, sequentially spanned
                v = torch.zeros(nbatch, ny)
                v[:, i] = 1
            else:
                # random properly-normalized vector for each sample
                v = self._random_vector(nbatch, ny)
            if x.is_cuda:
                v = v.cuda()
            vJ = jacobian_vector_product(y, x, v, create_graph=True)
            J2 += ny * torch.norm(vJ)**2 / (num_proj * nbatch)
        R = 1/2 * J2
        return R

    def _random_vector(self, nbatch, nv):
        '''
        creates a random vector of dimension C with a norm of C^(1/2)
        (as needed for the projection formula to work)
        '''
        if nv == 1: 
            return torch.ones(nbatch)
        v = torch.randn(nbatch, nv)
        arxilirary_zero = torch.zeros(nbatch, nv)
        vnorm = torch.norm(v, 2, -1, True)
        v = torch.addcdiv(arxilirary_zero, 1.0, v, vnorm)
        return v
                                                                            

class AutoDiff(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, y, x):

        nbatch_x, nx = x.shape
        nbatch_y, ny = y.shape
          
        assert nbatch_x == nbatch_y
        nbatch = nbatch_x
        
        J = torch.zeros(nbatch, ny, nx)
        for i in range(ny):
            # orthonormal vector, sequentially spanned
            v = torch.zeros(nbatch, ny)
            v[:, i] = 1

            if x.is_cuda:
                v = v.cuda()
            vJ = jacobian_vector_product(y, x, v, create_graph=True)
            if vJ is None:
                # dy/dx = 0
                break
            J[:, i, :] = vJ
            
        return J

class FiniteDiff(nn.Module):

    pass

                                                                            
    
