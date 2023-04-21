from torch import Tensor
import torch.nn as nn
import math
from functools import partial
from typing import Iterable
import torch
import torch.nn.functional as F

class Imod(nn.Module):
    '''
    A generalization of torch.nn.Identity()
    '''
    def __init__(self):
        super().__init__()

    def forward(self, *args):
        return args


class Seq(nn.ModuleList):
    '''
    A generalization of torch.nn.Sequential()
    '''
    def __init__(self, modules: Iterable[nn.Module]):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        x = self.__getitem__[0](*args, **kwargs)
        for i in range(1, self.__len__()):
            x = self.__getitem__[i](x)
        return x


class CosineCutoff(nn.Module):
    def __init__(self, rbound_upper=5.0):
        super().__init__()
        self.register_buffer("rbound_upper", torch.tensor(rbound_upper))

    def forward(self, distances):
        ru = self.rbound_upper
        rbounds = 0.5 * \
            (torch.cos(distances * math.pi / ru) + 1.0)
        rbounds = rbounds * (distances < ru).float()
        return rbounds



act_fn_dict = {
    "silu": partial(nn.SiLU, inplace=True),
    "relu": partial(nn.ReLU, inplace=True),
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "selu": partial(nn.SELU, inplace=True),
    "identity": nn.Identity
}
