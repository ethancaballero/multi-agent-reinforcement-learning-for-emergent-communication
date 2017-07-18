import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from util.prof_force.qrnn import QRNN_Model, QRNNLayer

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        try:
        	m.bias.data.fill_(0)
        except:
        	pass

class Discrim(torch.nn.Module):
    def __init__(self, args):
        super(Discrim, self).__init__()

        #self.gru = nn.GRUCell(self.feature_dim, 256)
        #self.gru = QRNNLayer(args.hidden_size, args.hidden_size)
        self.qrnn = QRNNLayer(args.num_agents*(args.hidden_size+args.max_vocab_size+args.num_agents-1), 256)
        #self.qrnn = QRNNLayer(256, 256)
        self.out_linear = nn.Linear(256, 1)

        self.apply(weights_init)
        self.out_linear.weight.data = normalized_columns_initializer(
            self.out_linear.weight.data, 1.0)
        self.out_linear.bias.data.fill_(0)

        #self.gru.bias_ih.data.fill_(0)
        #self.gru.bias_hh.data.fill_(0)

        self.train()

    def forward(self, x):
        x = self.qrnn(x)[:, -1, :]
        #x = self.out_linear(x)
        return F.sigmoid(self.out_linear(x))
