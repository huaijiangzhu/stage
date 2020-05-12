import torch
import torch.nn as nn
from stage.utils.nn import beye, bsm, flatten_non_batch

def cross_product_matrix(v):
    nb, dim = v.shape
    assert dim==3
    v1, v2, v3 = v[:, 0:1], v[:, 1:2], v[:, 2:3]
    v_hat = torch.cat((torch.zeros(nb, 1), -v3, v2, 
                       v3, torch.zeros(nb, 1), -v1, 
                       -v2, v1, torch.zeros(nb, 1)), dim=1)
    v_hat = v_hat.reshape(nb, 3, 3)
    return v_hat

def rodrigues_formula(w, th):
    if w.ndimension() == 1:
        w = w.unsqueeze(0)
    if th.ndimension() == 1:
        th = th.unsqueeze(0)
        th = th.reshape(-1, 1)
    nb, dim = w.shape
    assert dim==3
    w_hat = cross_product_matrix(w)
    w_norm = torch.norm(w, dim=1, keepdim=True)
    exp_wth = beye(nb, 3, 3)
    exp_wth += bsm(torch.sin(w_norm * th) / w_norm, w_hat)
    w_hat_squared = torch.bmm(w_hat, w_hat)
    exp_wth += bsm((1-torch.cos(w_norm * th)) / (w_norm ** 2), w_hat_squared)
    return exp_wth

def Rx(th):
    th = th.reshape(-1,1)
    nb, _ = th.shape
    axis = torch.Tensor([1,0,0]).repeat(nb, 1)
    return rodrigues_formula(axis, th)

def Ry(th):
    th.reshape(-1,1)
    nb, _ = th.shape
    axis = torch.Tensor([0,1,0]).repeat(nb, 1)
    return rodrigues_formula(axis, th)

def Rz(th):
    th.reshape(-1,1)
    nb, _ = th.shape
    axis = torch.Tensor([0,0,1]).repeat(nb, 1)
    return rodrigues_formula(axis, th)

def rpy_to_rotation_matrix(rpy):
    roll, pitch, yaw = rpy[:, 0:1], rpy[:, 1:2], rpy[:, 2:3]
    R = torch.bmm(Rz(yaw), torch.bmm(Ry(pitch), Rx(roll)))
    return R

class ForwardKinematics(nn.Module):
    '''
    Warning: be very careful with this, only works for fixed-base manipulator
    with revolute joints.
    '''
    
    def __init__(self, nq, joint_xyz, joint_rpy, joint_axis, link_xyz):
        super().__init__()
        self.nq, self.joint_axis = nq, joint_axis
        self.joints = self.joint_frame_chain(joint_xyz, joint_rpy)
        self.links = self.link_frame_chain(link_xyz)
        
    def forward(self, q, idx):
        nb, nq = q.shape[0:2]
        q = q.reshape(-1,1)
        axis = self.joint_axis.repeat(nb, 1)
        Tq = self.joint_rotation(axis, q)
        T = torch.bmm(self.joints.repeat(nb, 1, 1), Tq)
        T = T.reshape(nb, -1, 4, 4)
        T_world = beye(nb, 4, 4)
        for j in range(nq):
            T_world = torch.bmm(T_world, T[:, j, :, :])
        T_local = self.links[idx, :, :]
        T_local = T_local.unsqueeze(0).repeat(nb, 1, 1)
        T = torch.bmm(T_world, T_local)
        return T
        
    def joint_frame_chain(self, xyz, rpy):
        if xyz.ndimension() == 1:
            xyz = xyz.unsqueeze(0)
        if rpy.ndimension() == 1:
            rpy = rpy.unsqueeze(0)
        R = rpy_to_rotation_matrix(rpy)
        xyz = xyz.unsqueeze(-1)
        rpy = rpy.unsqueeze(-1)
        T = torch.cat((R, xyz), dim=-1)
        T = flatten_non_batch(T)
        nb = T.shape[0]
        last_row = torch.Tensor([0,0,0,1]).repeat(nb, 1)
        T = torch.cat((T, last_row), dim=-1).reshape(-1, 4, 4)
        return T
        
    def link_frame_chain(self, xyz, rpy=None):
        if rpy is None:
            rpy = torch.zeros_like(xyz)

        if xyz.ndimension() == 1:
            xyz = xyz.unsqueeze(0)
        if rpy.ndimension() == 1:
            rpy = rpy.unsqueeze(0)

        R = rpy_to_rotation_matrix(rpy)
        xyz = xyz.unsqueeze(-1)
        rpy = rpy.unsqueeze(-1)
        T = torch.cat((R, xyz), dim=-1)
        T = flatten_non_batch(T)
        nb = T.shape[0]
        last_row = torch.Tensor([0,0,0,1]).repeat(nb, 1)
        T = torch.cat((T, last_row), dim=-1).reshape(-1, 4, 4)
        return T
    
    def joint_rotation(self, axis, q):
        if axis.ndimension() == 1:
            axis = axis.unsqueeze(0)
        T = rodrigues_formula(axis, q)
        nb = T.shape[0]
        last_col = torch.Tensor([0,0,0]).reshape(-1,1).unsqueeze(0).repeat(nb, 1, 1)
        T = torch.cat((T, last_col), dim=-1)
        T = flatten_non_batch(T)
        last_row = torch.Tensor([0,0,0,1]).repeat(nb, 1)
        T = torch.cat((T, last_row), dim=-1).reshape(-1, 4, 4)
        return T
