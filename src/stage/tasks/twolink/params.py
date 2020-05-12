import torch
import numpy as np
from stage.utils.nn import use_gpu

use_gpu()
## kinematics params
JOINT_XYZ = torch.Tensor([[0, 0, 0.075], 
                          [1.05, 0, 0]])

JOINT_RPY = torch.Tensor([[0, -np.pi/2, 0], 
                          [0, 0, 0]])

JOINT_AXIS = torch.Tensor([[0, 1, 0], 
                          [0, 1, 0]])

LINK_XYZ = torch.Tensor([[0.5, 0, 0],
                         [0.5 , 0, 0]])