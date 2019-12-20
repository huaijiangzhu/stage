import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd

def use_gpu():
    torch.set_default_tensor_type(torch.cuda.FloatTensor if torch.cuda.is_available() 
                                                         else torch.FloatTensor)
def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def flatten_non_batch(x):
    return x.view(x.shape[0], -1)

def renew(x):
    return x.clone().detach()

def truncated_normal(shape, mean, std):
    tensor = torch.zeros(shape)
    tmp = tensor.new_empty(shape + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def get_affine_params(ensemble_size, in_features, out_features):
    shape = (ensemble_size, in_features, out_features)
    w = truncated_normal(shape=shape, mean=torch.zeros(shape),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)
    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features))

    return w, b

def swish(x):
    return x * torch.sigmoid(x)

def jacobian_vector_product(y, x, v, create_graph=False):
    # this computes v' * dy/dx
    flat_y = y.reshape(-1)
    flat_v = v.reshape(-1)
    vJ, = torch.autograd.grad(flat_y, x, flat_v, 
                              retain_graph=True,
                              allow_unused=True,
                              create_graph=create_graph)
 
    return vJ

def bmv(A, x):
    return A.bmm(x.unsqueeze(2)).squeeze(2)

def bdot(x, y):
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(1)

def bquad(Q, x):
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1)


