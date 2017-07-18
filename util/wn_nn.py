import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable

from functools import wraps
import math

from torch.nn.modules.rnn import RNNCellBase

def _decorate(forward, module, name, name_g, name_v):
    @wraps(forward)
    def decorated_forward(*args, **kwargs):
        g = module.__getattr__(name_g)
        v = module.__getattr__(name_v)
        w = v*(g/torch.norm(v)).expand_as(v)
        module.__setattr__(name, w)
        return forward(*args, **kwargs)
    return decorated_forward

def weight_norm(module, name, init=None):
    if init:
        module = init(module)

    param = module.__getattr__(name)

    # construct g,v such that w = g/||v|| * v
    g = torch.norm(param)
    v = param/g.expand_as(param)
    g = Parameter(g.data)
    v = Parameter(v.data)
    name_g = name + '_g'
    name_v = name + '_v'

    # remove w from parameter list
    del module._parameters[name]

    # add g and v as new parameters
    module.register_parameter(name_g, g)
    module.register_parameter(name_v, v)

    # construct w every time before forward is called
    module.forward = _decorate(module.forward, module, name, name_g, name_v)
    return module

def weight_norm_all(module, init=None):
    p_names = [n for n, p in module.named_parameters()]
    for n in p_names:
        if 'weight' in n:
            module = weight_norm(module, n, init)
    return module


class GRUCellSplitParams(RNNCellBase):
    """split the params so that weight_norm is applied separately for each matrix multiply"""
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCellSplitParams, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.weight_ih1 = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_ih2 = Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_ih3 = Parameter(torch.Tensor(hidden_size, input_size))

        self.weight_hh1 = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_hh2 = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_hh3 = Parameter(torch.Tensor(hidden_size, hidden_size))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(3 * hidden_size))
            self.bias_hh = Parameter(torch.Tensor(3 * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hx):
        self.weight_i = torch.cat((self.weight_ih1, self.weight_ih2, self.weight_ih3), 0)
        self.weight_h = torch.cat((self.weight_hh1, self.weight_hh2, self.weight_hh3), 0)
        return self._backend.GRUCell(
            input, hx,
            self.weight_i, self.weight_h,
            self.bias_ih, self.bias_hh,
        )