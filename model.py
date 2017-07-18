import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import util.gumbel_softmax as gumbel_softmax
import util.comm_util as comm_util

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import wraps

from util.wn_nn import weight_norm, weight_norm_all, GRUCellSplitParams

def _cnn_to_linear(seq, input_shape=None):
    if isinstance(input_shape, tuple):
        input_shape = list(input_shape)
    if input_shape is None:
        assert False, 'input_shape must be determined'
    for cnn in seq:
        if not isinstance(cnn, (nn.Conv2d, CReLU)):
            continue
        if isinstance(cnn, nn.Conv2d):
            kernel_size = cnn.kernel_size
            stride = cnn.stride
            padding = cnn.padding
            for i, l in enumerate(input_shape):
                input_shape[i] = (l - kernel_size[i] + stride[i] + 2*padding[i])//stride[i]
            channel_size = cnn.out_channels
        elif isinstance(cnn, CReLU):
            channel_size*=2
    return input_shape[0] * input_shape[1] * channel_size

class LinearEmbed(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = A_transpose*x + b`
    
    modification of linear that transposes self.weight
    so that tying encoder and decoder can be more easily applied
    as in https://arxiv.org/abs/1611.01462
    while still permitting gumbel softmax trick to be used
    """

    def __init__(self, in_features, out_features, bias=False):
        super(LinearEmbed, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        #self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, mode=None):


        if self.bias is None:
            return self._backend.Linear.apply(input, self.weight.t())
        else:
            return self._backend.Linear.apply(input, self.weight.t(), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= (torch.Tensor([[std]]) / torch.sqrt(out.pow(2).sum(1))).t()
    return out

def wn_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
        return m
    if classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        try:
            m.bias.data.fill_(0)
        except:
            pass
        return m

def weights_init(m):
    classname = m.__class__.__name__
    #'''
    if classname.find('Conv') != -1:
        try:
            weight_shape = list(m.weight.data.size())
            fan_in = np.prod(weight_shape[1:4])
            fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
        except:
            pass
        try:
            m.bias.data.fill_(0)
        except:
            pass
        #'''
    elif classname.find('Linear') != -1:
        try:
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
        except:
            pass
        try:
            m.bias.data.fill_(0)
        except:
            pass

#https://github.com/pytorch/pytorch/issues/1327
class CReLU(nn.ReLU):
    def __init__(self):
        super(CReLU, self).__init__()
    def forward(self, input):
        return torch.cat(
            (F.relu(input, self.inplace), 
            F.relu(-input, self.inplace)), 
            1)

class Comm(torch.nn.Module):

    def __init__(self, ob_space__omni, ob_space__all, action_space__all, action_space_default, args):
    #def __init__(self, num_inputs, action_space, args):    
        super(Comm, self).__init__()
        self.args = args
        self.agents = nn.ModuleList([agent(i, ob_space__all[i], action_space__all[i], action_space_default, args) for i in range(args.num_agents)])
        
        self.omni_critic = omni_critic(ob_space__omni, args)

        # https://arxiv.org/abs/1611.01462
        if args.tie_enc_dec_weights:
            _1st_agent_w_msg_embed = [adx for adx in range(len(self.agents)) if adx == comm_util.single_actor_settings(args)][0]
            if args.hidden_size == args.enc_size: #<--TODO: might be able to bypass check if extra proj layer is added to proj to hid size (or vice versa)
                for a in self.agents:
                    for idx in range(args.num_agents-1):
                        a.msg_linear_embed.weight = self.agents[_1st_agent_w_msg_embed].msg_linear_embed.weight
                        a.msg_linear.weight = self.agents[_1st_agent_w_msg_embed].msg_linear_embed.weight
        self.train()

    def forward(self, omni_state, omni_features, inputs__all):
        #critics=[]
        actors=[]
        msg_recps=[]
        msgs=[]
        features=[]
        msg_embeds=[]

        for adx, a in enumerate(self.agents):
            agent_return = a(adx, inputs__all[adx])
            #critics.append(agent_return[0])
            actors.append(agent_return[1])
            msg_recps.append(agent_return[2])
            msgs.append(agent_return[3])
            features.append(agent_return[4])
            msg_embeds.append(agent_return[5])

        critics_tensor = self.omni_critic([omni_state, omni_features, msg_embeds[comm_util.single_actor_settings(self.args)]])
        critics = list(torch.split(critics_tensor[0], 1, -1))
        critic_features = critics_tensor[1]

        return critics, actors, msg_recps, msgs, features, critic_features

class omni_critic(torch.nn.Module):
    #def __init__(self, obs_space, action_space):
    def __init__(self, ob_space__omni, args):
        super(omni_critic, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(ob_space__omni[0], 16, 3, stride=2, padding=1),
            CReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            CReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            CReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            CReLU(),
            )

        #'''TODO: try version with and without critic seeing messages (and detached at different spots)'''

        self.msg_linear_embed = LinearEmbed(args.max_vocab_size, args.enc_size, False)
        self.feature_dim = _cnn_to_linear(self.f, ob_space__omni[1:])
        #self.lstm = nn.LSTMCell(self.feature_dim + self.enc_size, self.hid_size)
        self.lstm = nn.LSTMCell(self.feature_dim, args.hidden_size)

        self.apply(weights_init)
        #self.critic = nn.Linear(args.hidden_size*2*args.num_agents, args.num_agents)
        self.critic = nn.Linear(args.hidden_size, args.num_agents)
        self.critic.weight.data = normalized_columns_initializer(
                self.critic.weight.data, 1.0)
        self.critic.bias.data.fill_(0)

    def forward(self, inputs):       
        ob, (hx, cx), msg_embed = inputs
        x = self.f(ob)
        x = x.view(-1, self.feature_dim)

        hx = self.lstm(x, (hx, cx))
        x = hx[0]
        return self.critic(x), hx

class agent(torch.nn.Module):
    #def __init__(self, obs_space, action_space):
    def __init__(self, id, obs_space, action_space, action_space_default, args):
        super(agent, self).__init__()

        self.f = nn.Sequential(
            nn.Conv2d(obs_space[0], 16, 3, stride=2, padding=1),
            CReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            CReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            CReLU(),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            CReLU(),
            )

        self.args = args
        self.hid_size = args.hidden_size
        self.enc_size = args.enc_size
        self.feature_dim = _cnn_to_linear(self.f, obs_space[1:])
        #self.feature_dim = self.get_flat_fts(obs_space, self.f)
        self.max_vocab_size = args.max_vocab_size

        #self.dropout = nn.Dropout(p=args.dropout)
        if id == comm_util.single_actor_settings(args):
            self.lstm = nn.LSTMCell(self.feature_dim + self.enc_size, self.hid_size)
        else:
            self.lstm = nn.LSTMCell(self.feature_dim + self.enc_size + args.num_agents-1, self.hid_size)

        '''TODO: figure out way to combine multiple recv msgs'''
        #self.msg_linear_embeds = nn.ModuleList([LinearEmbed(args.max_vocab_size, self.enc_size, False) for _ in range(args.num_agents-1)])
        self.msg_linear_embed = LinearEmbed(args.max_vocab_size, self.enc_size, False)

        self.num_outputs = action_space

        self.critic_linear = nn.Linear(self.hid_size, 1)
        self.actor_linear = nn.Linear(self.hid_size, self.num_outputs)

        '''TODO: 1 shared msg_sender with an ID to figure out how to send based on ID of agent that will recv msg'''
        #self.msg_linears = nn.ModuleList([nn.Linear(self.hid_size, args.max_vocab_size, False) for _ in range(args.num_agents-1)])
        self.msg_linear = nn.Linear(self.hid_size, args.max_vocab_size, False)
        self.msg_recp_select = nn.Linear(self.hid_size, args.num_agents-1, False)
        
        self.dummy_hot = torch.FloatTensor(args.max_vocab_size)

        self.apply(weights_init)
        if id == comm_util.single_actor_settings(args):
            self.actor_linear.weight.data = normalized_columns_initializer(
                self.actor_linear.weight.data, 0.01)
            self.actor_linear.bias.data.fill_(0)
            self.critic_linear.weight.data = normalized_columns_initializer(
                self.critic_linear.weight.data, 1.0)
            self.critic_linear.bias.data.fill_(0)

        try:
            self.lstm.bias_ih.data.fill_(0)
            self.lstm.bias_hh.data.fill_(0)
        except:
            self.lstm.bias.data.fill_(0)

        try:
            self.msg_linear.bias.data.fill_(0)
            self.msg_recp_select.bias.data.fill_(0)
        except:
            pass

        self.train()

    def get_flat_fts(self, in_size, fts):
        f = fts(Variable(torch.ones(1,*in_size)))
        return int(np.prod(f.size()[1:]))

    def forward(self, id, inputs):
        ob, msg_recp, [msg_prev_self, msg_recv_split], (hx, cx) = inputs
        x = self.f(ob)
        x = x.view(-1, self.feature_dim)

        msg_prev_self_embed = self.msg_linear_embed(msg_prev_self)

        msg_embed_split = [torch.squeeze(self.msg_linear_embed(msg_recv)) for idx, msg_recv in enumerate(msg_recv_split)]

        msg_embed = torch.stack(msg_embed_split, dim=-1)
        msg_embed = msg_embed.view(1, -1)

        if id == comm_util.single_actor_settings(self.args):
            '''
            if x.size()[1]/msg_embed.size()[1] > 1:
                msg_embed = msg_embed.repeat(1,math.ceil(x.size()[1]/msg_embed.size()[1])).resize_as(x)
            elif x.size()[1]/msg_embed.size()[1] < 1:
                #print(x.size(), msg_embed.size())
                #print()
                x = x.repeat(1,math.ceil(msg_embed.size()[1]/x.size()[1])).resize_as(msg_embed)
            else:
                pass
                #'''
            x = torch.cat([x, msg_embed], x.dim()-1)
            #x = torch.cat([x, msg_prev_self_embed, msg_embed, msg_recp], x.dim()-1)

        else:
            #x = torch.cat([x, msg_prev_self_embed, msg_embed, msg_recp], x.dim()-1)
            x = torch.cat([x, msg_prev_self_embed, msg_recp], x.dim()-1)

            pass

        hx = self.lstm(x, (hx, cx))
        x = hx[0]

        hard=True
        hi_def=self.args.avg_gumbel

        return None, self.actor_linear(x), F.sigmoid(self.msg_recp_select(x)), \
            gumbel_softmax.gumbel_softmax_sample(self.msg_linear(x), self.dummy_hot, hard, hi_def, is_training=self.training), \
            hx, \
            msg_embed


