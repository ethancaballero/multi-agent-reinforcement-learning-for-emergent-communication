# Based off code from this blog post:
# https://metamind.io/research/new-neural-network-building-block-allows-faster-and-more-accurate-text-understanding/

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np

class Linear(nn.Linear):

    def forward(self, x):
        shape = x.size()
        if len(shape) == 3:
            x = x.view(-1, shape[2])
        y = super(Linear, self).forward(x)
        if len(shape) == 3:
            y = y.view(shape[0], shape[1], y.size(1))
        return y


def strnn(f, z, hinit):
    batch_size, seq_length, model_dim = f.size()
    h = [hinit]

    # Kernel
    for t in range(seq_length):
        prev_h = h[-1]
        ft = f[:, t, :]
        zt = z[:, t, :]

        ht = prev_h * ft + zt
        h.append(ht)

    hs = torch.cat([hh.unsqueeze(1) for hh in h[1:]], 1)
    return hs


def reverse_tensor(var, dim):
    dim_size = var.size(dim)
    index = [i for i in range(dim_size - 1, -1, -1)]
    index = torch.LongTensor(index)
    if isinstance(var, Variable):
        index = to_gpu(Variable(index, volatile=var.volatile))
    inverted_tensor = var.index_select(dim, index)
    return inverted_tensor


class QRNN_Model(nn.Module):
    def __init__(self, inp_dim=None, model_dim=None, num_layers=None, mlp_dim=None, num_classes=None,
                 dropout_rate=0.5, kernel_size=None,
                 **kwargs):
        super(Model, self).__init__()

        self.init_qrnn(
            inp_dim=inp_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            )
        self.l0 = nn.Linear(model_dim, mlp_dim)
        self.l1 = nn.Linear(mlp_dim, mlp_dim)
        self.l2 = nn.Linear(mlp_dim, num_classes)
        self.dropout_rate = dropout_rate

    def init_qrnn(self, inp_dim, model_dim, num_layers, kernel_size):
        self.num_layers = num_layers
        x_dim = inp_dim

        for i in range(num_layers):
            setattr(self, "qrnn_{}".format(i), QRNNLayer(
                in_size=x_dim,
                out_size=model_dim,
                kernel_size=kernel_size,
                ))
            x_dim = model_dim

    def run_qrnn(self, x):
        for i in range(self.num_layers):
            x = getattr(self, "qrnn_{}".format(i))(x)
        return x

    def forward(self, x):
        #print("x_all", x.size())
        qs = self.run_qrnn(x)
        q = qs[:, -1, :]
        q = F.relu(F.dropout(self.l0(q), self.dropout_rate, self.training))
        q = F.relu(F.dropout(self.l1(q), self.dropout_rate, self.training))
        q = self.l2(q)
        return q


class QRNNLayer(nn.Module):

    def __init__(self, in_size, out_size, kernel_size=2):
        super(QRNNLayer, self).__init__()
        if kernel_size == 1:
            self.W = Linear(in_size, 3 * out_size)
        elif kernel_size == 2:
            self.W = Linear(in_size, 3 * out_size, bias=False)
            self.V = Linear(in_size, 3 * out_size)
        else:
            self.conv = nn.Conv1d(in_size, 3 * out_size, kernel_size,
                                     stride=1, padding=kernel_size - 1)
        self.in_size, self.size = in_size, out_size
        self.kernel_size = kernel_size

    def pre(self, x):
        if self.kernel_size == 1:
            ret = self.W(x)
        elif self.kernel_size == 2:
            xprev = Variable(torch.from_numpy(
                np.zeros((self.batch_size, 1, self.in_size),
                              dtype=np.float32)), volatile=not self.training)
            xtminus1 = torch.cat((xprev, x[:, :-1, :]), 1)
            ret = self.W(x) + self.V(xtminus1)
        else:
            ret = self.conv(x.transpose(1,2).contiguous()).transpose(1,2).contiguous()

        return ret

    def reset_state(self):
        self.c = Variable(torch.from_numpy(np.zeros((self.batch_size, self.size),
                                            dtype=np.float32)), volatile=not self.training)

    def forward(self, x):
        #print("x_layer", x.size())
        #asdf
        self.batch_size = x.size()[0]
        self.reset_state()

        dims = len(x.size()) - 1
        f, z, o = torch.chunk(self.pre(x), 3, dims)
        f = F.sigmoid(f)
        z = (1 - f) * F.tanh(z)
        o = F.sigmoid(o)

        self.c = strnn(f, z, self.c[:self.batch_size])
        self.h = self.c * o

        return self.h
