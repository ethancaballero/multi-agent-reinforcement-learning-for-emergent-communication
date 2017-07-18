from __future__ import print_function

import argparse
import os
import sys
import time, random

import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import reduce

from model import Comm
from train_uni import fwd, bwd, replay_server
from train_pause import train_pause
from test import test
from util.replay_buffer import ReplayBuffer
import util.my_optim as my_optim
from env_util.env_spec_proc import get_env_spec
from util.comm_util import *

from tensorboard_logger import configure, log_value

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num_processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')

parser.add_argument('--max_episode_length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env_name', default='BreakoutDeterministic-v4', metavar='ENV',
                    help='environment to train on (default: BreakoutDeterministic-v4)')
parser.add_argument('--no_shared', default=False, metavar='O',
                    help='use an optimizer without shared momentum.')

parser.add_argument('--tau', type=float, default=1e-2)

parser.add_argument('--num_steps', type=int, default=27, metavar='NS', help='number of forward steps in PCL episode (default: 20)')
parser.add_argument('--rollout_len', type=int, default=27, help='number of forward steps in PCL rollout scans (default: 20)')

parser.add_argument('--loss', default='a3c', metavar='l', help='loss used')

parser.add_argument('--pi_loss_coef', type=int, default=1.0)
parser.add_argument('--v_loss_coef', type=int, default=0.5)
parser.add_argument('--normalize_loss_by_steps', type=str2bool, default=True)

parser.add_argument('--onp', type=str2bool, default=True)
parser.add_argument('--offp', type=str2bool, default=False)
parser.add_argument('--off_policy_rate', type=int, default=1)
parser.add_argument('--bwd2fwd_rate', type=int, default=1)
#parser.add_argument('--batchsize', type=int, default=20)
parser.add_argument('--replay_buffer_size', type=int, default=5000)
parser.add_argument('--uni_log', type=str2bool, default=False)

parser.add_argument('--debug_vision', type=str2bool, default=False)
parser.add_argument('--entropy_record', type=str2bool, default=True)

parser.add_argument('--train', type=str2bool, default=True)
parser.add_argument('--test', type=str2bool, default=True)
parser.add_argument('--test_sleep', type=str2bool, default=False)

#parser.add_argument('--save', type=str2bool, default=True)
parser.add_argument('--save', type=str2bool, default=False)
parser.add_argument('--load', type=str2bool, default=False)
parser.add_argument('--save_interval', type=int, default=300, help='seconds between saving')

parser.add_argument('--save_file', type=str, default="model")
parser.add_argument('--load_file', type=str, default="model")

parser.add_argument('--survival_reward', type=int, default=0)
parser.add_argument('--death_penalty', type=int, default=0)
parser.add_argument('--retry_recurse', type=str2bool, default=False, help='restart env if it crashes; use with uni flash games')

parser.add_argument('--train_pause', type=str2bool, default=True, help='allowed to pause fwd to perform bwd')


parser.add_argument('--num_agents', type=int, default=2, metavar='a', help='number of agents')
parser.add_argument('--vocab_id', default="basic-english", help='which vocab to use')
parser.add_argument('--max_vocab_size', default=40, type=int, help='max vocab size, default (-1) means no max')

parser.add_argument('--hidden_size', default=320, type=int, help='size of hidden states')
parser.add_argument('--enc_size', default=320, type=int, help='size of msg encoder states')
parser.add_argument('--tie_enc_dec_weights', type=str2bool, default=True, help='https://arxiv.org/abs/1611.01462')
parser.add_argument('--dropout', default=0.0, type=float)

parser.add_argument('--trpo', type=str2bool, default=False, help='trpo (1st order avg version from https://arxiv.org/abs/1611.01224)')
parser.add_argument('--alpha', type=float, default=.99, help='alpha for TRPO')
parser.add_argument('--trust_region_delta', type=float, default=1, help='trust_region_delta for TRPO')

parser.add_argument('--dirichlet_vocab', type=str2bool, default=True, help='dirichlet vocab process on/off')
parser.add_argument('--d_alpha', type=float, default=10000, help='alpha for dirichlet vocab process')
parser.add_argument('--d_weight', type=float, default=.25, help='how large are dirichlet vocab process rewards')
parser.add_argument('--utter_penalty', type=float, default=1.1, help='reward of each utterance that is not 0 idx is divided by this')

parser.add_argument('--throughput_log', type=str2bool, default=False, help='log server throughput')

parser.add_argument('--onp_cache', type=str2bool, default=False, help='onp_cache bool')

parser.add_argument('--pf', type=str2bool, default=False, help='professor forcing: https://arxiv.org/abs/1610.09038')
parser.add_argument('--pf_weight', type=int, default=0.01, help='how much professor forcing contributes to loss')
parser.add_argument('--d_acc_avg_size', type=int, default=100, help='how many values discrim accuracy run_avg consists of')

parser.add_argument('--avg_gumbel', type=str2bool, default=True, help='average over N gumbel dists')
parser.add_argument('--opt_eps', type=float, default=1e-3, help='how large is eps in optimizer denominator')


parser.add_argument('--demonstrate', type=str2bool, default=False)

if __name__ == '__main__':

    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENAI_REMOTE_VERBOSE'] = '0'

    args = parser.parse_args()
    if args.demonstrate:
        configure("tmp/tb_log"+"_"+str(args.load_file)+"_demo")
    else:
        configure("tmp/tb_log")

    log_value("h_params/torch___"+torch.__version__, 0)
    for h_param in args.__dict__:
        log_value("h_params/"+h_param+"___"+str(getattr(args, h_param)), 0)

    kwargs = {"multi_obs_settings": multi_obs_settings(args)}

    if args.retry_recurse==True:
        def retry_recurse(func, _args, **_):
            try:
                func(*_args, **_)
            except:
                pass
            log_value('crash/rank_crashed', _args[0]+random.randint(1, 50)/100)
            retry_recurse(func, _args, **_)
    else:
        def retry_recurse(func, _args, **_):
            func(*_args, **_)

    torch.manual_seed(args.seed)

    dirichlet_vocab = DirichletVocab(args)

    # This is to bypass importing universe in main process (twisted would break mp)
    env_spec_q = mp.Queue(1)
    env_spec_p = mp.Process(target=get_env_spec, args=(args, env_spec_q), kwargs=kwargs)
    env_spec_p.start()
    observation_space_omni, observation_space, action_space = env_spec_q.get()
    env_spec_q.close()
    env_spec_p.terminate()

    ob_space__omni = observation_space_omni.shape

    #ob_space__all = [_state.shape for _state in env_multi_splitter(np.zeros(observation_space.shape), *mask_settings(args))]
    ob_space__all = [ob.shape for ob in observation_space]

    #HACK
    action_space__all = [action_space.n for _ in range(args.num_agents)]

    shared_model = Comm(ob_space__omni, ob_space__all, action_space__all, action_space.n, args)
    shared_model.share_memory()
    if args.load:
        shared_model.load_state_dict(torch.load('./weights/'+args.load_file+'.pth'))
    shared_model_avg=None
    if args.trpo:
        shared_model_avg = Comm(ob_space__omni, ob_space__all, action_space__all, action_space.n, args)
        shared_model_avg.load_state_dict(shared_model.state_dict())
        shared_model_avg.share_memory()
        for param in shared_model_avg.parameters():
            param.requires_grad = False
    shared_discrim_model=None
    d_acc_shared=None
    if args.pf:
        from util.prof_force.discrim import Discrim
        shared_discrim_model = Discrim(args)
        shared_discrim_model.share_memory()
        d_acc_shared = Variable(torch.FloatTensor(1).share_memory_().zero_(), requires_grad=False)
        if args.load:
            shared_discrim_model.load_state_dict(torch.load('./weights/d_'+args.load_file+'.pth'))

    '''
    if args.load == True:
        if args.trpo:
            shared_model_avg.load_state_dict(torch.load('./weights/'+args.load_file+'_avg'+'.pth'))
            #'''
    print(shared_model)
    total_params = sum([reduce(lambda x, y: x * y, p.size()) for p in shared_model.parameters()])
    print(total_params)
    if args.pf:
        print(shared_discrim_model)
        total_d_params = sum([reduce(lambda x, y: x * y, p.size()) for p in shared_discrim_model.parameters()])
        print(total_d_params)
        log_value("h_params/total_d_params", total_d_params)
    log_value("h_params/total_params", total_params)

    processes = []

    if args.test:
        # hardcode client_id to 1 instead of rank so that test agent is in 1st VNC window
        p = mp.Process(target=retry_recurse, args=(test, (1, args, shared_model, dirichlet_vocab)), kwargs=kwargs)
        p.start()
        processes.append(p)

    if args.train:
        if args.no_shared:
            optimizer = None
            d_optimizer = None
        else:
            optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr, eps=args.opt_eps)
            optimizer.share_memory()
            d_optimizer = None
            if args.pf:
                d_optimizer = my_optim.SharedAdam(shared_discrim_model.parameters(), lr=args.lr, eps=args.opt_eps)
                d_optimizer.share_memory()

        bwd_count = Variable(torch.LongTensor(1).share_memory_().zero_(), requires_grad=False)

        #universe (unpausable) envs
        if not args.train_pause:

            fwd_count = Variable(torch.LongTensor(1).share_memory_().zero_(), requires_grad=False)
            '''TODO: should you use multiprocessing.Manager().Queue() ?'''
            replay_buffer = ReplayBuffer(args)
            #fwd_q = mp.Queue(5*args.off_policy_rate*args.num_processes)
            #bwd_q = mp.Queue(5*args.off_policy_rate*args.num_processes*args.bwd2fwd_rate)
            fwd_q = mp.Manager().Queue(5*args.off_policy_rate*args.num_processes)
            bwd_q = mp.Manager().Queue(5*args.off_policy_rate*args.num_processes*args.bwd2fwd_rate)
            p = mp.Process(target=replay_server, args=(args, replay_buffer, fwd_q, bwd_q, fwd_count, bwd_count))
            p.start()
            processes.append(p)

            env_details = (ob_space__omni, ob_space__all, action_space__all)

            # +2 so that test agent (hopefully) appears in first VNC screen
            for rank in range(0+2, args.num_processes+2):
                p = mp.Process(target=retry_recurse, args=(fwd, (rank, args, shared_model, fwd_q, fwd_count)), kwargs=kwargs)
                p.start()
                processes.append(p)
            for rank in range(0+2, (args.num_processes*args.bwd2fwd_rate)+2):
                p = mp.Process(target=bwd, args=(rank, args, shared_model, shared_model_avg, shared_discrim_model, bwd_q, env_details, dirichlet_vocab, bwd_count, d_acc_shared, optimizer, d_optimizer), kwargs={})
                p.start()
                processes.append(p)

        # non-universe (pausable) envs
        else:
            for rank in range(0+2, args.num_processes+2):
                p = mp.Process(target=train_pause, args=(rank, args, shared_model, shared_model_avg, dirichlet_vocab, bwd_count, optimizer), kwargs=kwargs)
                p.start()
                processes.append(p)

    for p in processes:
        p.join()

    #why did uni starter agent have this here:
    #while True:
        #time.sleep(1000)
