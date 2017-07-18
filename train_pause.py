import math
import os
import sys
import time
import random
from collections import deque

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Comm
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from env_util.env_fixes import init_actions, state_stuck

from util.comm_util import *

from util.loss_util import pcl_loss, a3c_loss, loss_with_kl_constraint, discrim_loss

from tensorboard_logger import configure, log_value

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def update_avg(shared_model_avg, model, alpha):
    for shared_avg_param, param in zip(shared_model_avg.parameters(), model.parameters()):
        shared_avg_param.data = shared_avg_param.data*alpha.expand_as(shared_avg_param) + (1-alpha).expand_as(param)*param.data

#apply op to all agents in list
def _a(a_list, op=None):
    if op!=None:
        return [op(a)for a in a_list]
    else:
        return [a for a in a_list]

def n_list_to_t_list(a_list):
    return [torch.from_numpy(_a_list) for _a_list in a_list]

def train_pause(rank, args, shared_model, shared_model_avg, dirichlet_vocab, bwd_count, optimizer=None, **kwargs):
    from env_util.envs import create_env
    torch.manual_seed(args.seed + rank)

    env = create_env(args.env_name, rank, 1, **kwargs)
    env.seed([args.seed + rank])

    ob_space__all = [ob.shape for ob in env.observation_space]

    #HACK
    action_space__all = [env.action_space.n for _ in range(args.num_agents)]

    #model = Comm(env.observation_space_omni.shape, action_space__all, env.action_space.n, args)
    model = Comm(env.observation_space_omni.shape, ob_space__all, action_space__all, env.action_space.n, args)
    model.train()

    state = env.reset()
    state = env_from_numpy(state)
    done = True

    episode_length = 0
    reward_sum = 0
    action_init = init_actions(args.env_name)
    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    env_init = torch_init = rank
        
    entropy_sum = 0
    dirichlet_reward_sum = 0
    if rank == 2:
        word_used_bools = [0 for _ in range(args.max_vocab_size)]
    while True:
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            msg = [Variable(torch.zeros(1, args.max_vocab_size)) for _ in range(args.num_agents)]
            msg_recp = [Variable(torch.zeros(1, args.num_agents-1)) for _ in range(args.num_agents)]
            #features = [Variable(torch.zeros(1, args.hidden_size)) for _ in range(args.num_agents)]
            features = [(Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))) for _ in range(args.num_agents)]
            features_omni = [Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))]
        else:
            msg = [Variable(_msg.data) for _msg in msg]
            msg_recp = [Variable(_msg_recp.data) for _msg_recp in msg_recp]
            #features = [Variable(_feature.data) for _feature in features]
            features = [(Variable(_f.data) for _f in _feature) for _feature in features]
            features_omni = [Variable(torch.zeros(1, args.hidden_size)), Variable(torch.zeros(1, args.hidden_size))]

        values = []
        log_probs = []
        rewards = []
        entropies = []

        if args.dirichlet_vocab:
            dirichlet_reward = 0

        if args.trpo:
            probs_avg = []
            log_probs_dist = []

        '''TODO: add randomness to num_steps to promote path consistency for multiple lengths'''
        #for step in range(args.num_steps + random.randrange(0,21,20)):
        for step in range(args.num_steps):
            episode_length += 1

            msg_recv = swap_msgs(msg, msg_recp)
            value, logit, msg_recp, msg, features, features_omni = model(Variable(state[0].float().unsqueeze(0)), features_omni, [(Variable(_state.float().unsqueeze(0)), _msg_recp, _msg_recv, _features) \
                        for _state, _msg_recp, _msg_recv, _features in zip(state[1], msg_recp, msg_recv, features)])

            if rank==2:
                for mdx, _msg in enumerate(msg):
                    '''TODO: add msg_recp gate'''
                    msg_index = int(torch.max(_msg, 1)[1].data.numpy()[0])
                    for recp in range(args.num_agents-1):
                        if recp == mdx:
                            recp += 1
                        else: 
                            pass
                        word_used_bools[msg_index] = 1

            prob = [F.softmax(_logit).clamp(max=1 - 1e-20) for _logit in logit]
            action = [_prob.multinomial().data for _prob in prob]
            log_prob_dist = [_prob.log() for _prob in prob]

            #TODO: multiple nonlingustic actions
            #entropy = [-(_log_prob_dist * _prob).sum(1) for _log_prob_dist in log_prob_dist]
            entropy = -(log_prob_dist[single_actor_settings(args)] * prob[single_actor_settings(args)]).sum(1)
            entropies.append(entropy)
            log_prob = [_log_prob_dist.gather(1, Variable(_action)) for _log_prob_dist, _action in zip(log_prob_dist, action)]

            if args.entropy_record and rank==2:
                #TODO: only agent at specified index performs non-linguistic actions for this; need to undo later
                #entropy += sum(-(_log_prob * _prob).sum(1).data.numpy()[0] for _log_prob, _prob in zip(log_prob_dist, prob))
                entropy_sum += entropy

            #'''#TODO: only agent at specified index performs non-linguistic actions for this; need to undo later
            action = action[single_actor_settings(args)]
            #'''

            if action_init and (episode_length < len(action_init)):
                state, reward, done, info = env.step([action_init[episode_length]])
            else:
                state, reward, done, _ = env.step(action.numpy()[0])

            #env.render()

            done = done[0] or episode_length >= args.max_episode_length or state_stuck(args.env_name, state)
            reward = sum([max(min(_reward, 1), -1) for _reward in reward])

            if args.dirichlet_vocab:
                _dirichlet_reward = (dirichlet_vocab.reward(msg)*args.d_weight)
                reward += _dirichlet_reward/args.num_steps
                dirichlet_reward_sum += _dirichlet_reward
                dirichlet_vocab.update(msg)

            reward_sum += reward

            if done:
                torch_init = rank*1000+random.randint(1,999)
                env_init = rank*1000+random.randint(1,999)
                torch.manual_seed(args.seed + torch_init)
                env.seed([args.seed + env_init])
                state = env.reset()

                # TODO: intrinsic fear https://arxiv.org/pdf/1611.01211.pdf
                # hack to get agent to learn not to die
                if episode_length < args.max_episode_length:
                    reward += -args.death_penalty
                    reward_sum += -args.death_penalty

                if rank == 2:
                    log_value('train'+str(rank)+'/episode_length', episode_length)
                    log_value('train'+str(rank)+'/reward_sum', reward_sum)
                    if args.entropy_record:
                        log_value('train'+str(rank)+'/entropy', entropy_sum.data.numpy()[0]/episode_length)
                    if args.dirichlet_vocab:
                        log_value('train'+str(rank)+'/dirichlet_reward', dirichlet_reward_sum/episode_length)
                    log_value('train'+str(rank)+'/unique_words_used', sum(word_used_bools))
                    word_used_bools = [0 for _ in range(args.max_vocab_size)]
                episode_length = 0
                reward_sum = 0
                dirichlet_reward_sum = 0
                entropy_sum = 0

            state = env_from_numpy(state)
            values += [value]
            log_probs += [log_prob]
            rewards += [reward]
            
            if done:
                break

        R = [torch.zeros(1, 1) for _ in range(args.num_agents)]
        if not done:
            msg_recv = swap_msgs(msg, msg_recp)
            value, _, _, _, _, _ = model(Variable(state[0].float().unsqueeze(0)), features_omni, [(Variable(_state.float().unsqueeze(0)), _msg_recp, _msg_recv, _feature) \
                        for _state, _msg_recp, _msg_recv, _feature in zip(state[1], msg_recp, msg_recv, features)])
            R = [_value.data for _value in value]

        R = [Variable(_R) for _R in R]

        if len(rewards) > 0:

            # hack to get agent to learn to survive
            rewards[-1] += args.survival_reward
            reward_sum += args.survival_reward

            if not action_init:
                if args.loss == 'pcl':
                    pi_loss, v_loss = pcl_loss(args, rewards, values, log_probs, R)
                elif args.loss == 'a3c':
                    pi_loss, v_loss = a3c_loss(args, rewards, values, log_probs, entropies, R)
                if args.trpo:
                    pi_loss, kl = loss_with_kl_constraint(
                        probs_avg,
                        log_probs_dist,
                        model,
                        pi_loss,
                        args.trust_region_delta,
                        optimizer,
                        args
                        )

                loss = pi_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), 40)
                ensure_shared_grads(model, shared_model)
                optimizer.step()

                if args.trpo:
                    update_avg(shared_model_avg, model, alpha)

                if args.throughput_log:
                    bwd_count += len(rewards)
