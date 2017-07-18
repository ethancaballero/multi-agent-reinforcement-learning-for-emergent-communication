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



class PartialRollout(object):
    """
    a piece of a complete rollout.  We run our agent, and process its experience
    once it has processed enough steps.
    """
    def __init__(self, rank):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.R = 0.0
        self.terminal = False
        self.features = []
        self.features_omni = []
        self.msgs = []
        self.msg_recps = []

        self.probs = []
        self.rank = rank

    #def add(self, state, action, reward, value, terminal, features):
    def add(self, state, action, reward, value, terminal, features, features_omni, probs):
        self.states += [state]
        self.actions += [action]
        self.rewards += [reward]
        self.values += [value]
        self.terminal = terminal

        self.features += [features]
        self.features_omni += [features_omni]
        self.msgs += [msgs]
        self.msg_recps += [msg_recps]

        self.probs += [prob]

    def extend(self, other):
        assert not self.terminal
        self.states.extend(other.states)
        self.actions.extend(other.actions)
        self.rewards.extend(other.rewards)
        self.values.extend(other.values)
        self.R = other.R
        self.terminal = other.terminal
        '''# you always just use the first feature
        self.features.extend(other.features)
        self.features_omni.extend(other.features_omni)
        self.msgs.extend(other.msgs)
        self.msg_recps.extend(other.msg_recps)
        #'''

        self.probs.extend(other.probs)

def env_runner(rank, args, env, model, shared_model, fwd_count):
    state = env.reset()
    done = True

    episode_length = 0
    reward_sum = 0
    action_init = init_actions(args.env_name)
    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    env_init = torch_init = rank
    while True:
        roll = PartialRollout(torch_init)
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            msg = [Variable(torch.zeros(1, args.max_vocab_size), volatile=True) for _ in range(args.num_agents)]
            msg_recp = [Variable(torch.zeros(1, args.num_agents-1), volatile=True) for _ in range(args.num_agents)]
            features = [(Variable(torch.zeros(1, args.hidden_size), volatile=True), Variable(torch.zeros(1, args.hidden_size), volatile=True)) for _ in range(args.num_agents)]
            features_omni = [Variable(torch.zeros(1, args.hidden_size), volatile=True), Variable(torch.zeros(1, args.hidden_size), volatile=True)]

        else:
            msg = [Variable(_msg.data, volatile=True) for _msg in msg]
            msg_recp = [Variable(_msg_recp.data, volatile=True) for _msg_recp in msg_recp]
            features = [[Variable(_f.data, volatile=True) for _f in _feature] for _feature in features]
            features_omni = [Variable(_f.data, volatile=True) for _f in features_omni]

        roll.msgs += [[_msg.data.numpy() for _msg in msg]]
        roll.msg_recps += [[_msg_recp.data.numpy() for _msg_recp in msg_recp]]
        roll.features += [[[_f.data.numpy() for _f in _feature] for _feature in features]]
        roll.features_omni += [[_feature.data.numpy() for _feature in features_omni]]

        '''TODO: add randomness to num_steps to promote path consistency for multiple lengths'''
        #for step in range(args.num_steps + random.randrange(0,21,20)):
        for step in range(args.num_steps):
            episode_length += 1
            roll.states += [state]
            #state = [torch.from_numpy(_state) for _state in state]
            state = env_from_numpy(state)

            msg_recv = swap_msgs(msg, msg_recp)
            value, logit, msg_recp, msg, features, features_omni = model(Variable(state[0].float().unsqueeze(0), volatile=True), features_omni, [(Variable(_state.float().unsqueeze(0), volatile=True), _msg_recp, _msg_recv, _features) \
                        for _state, _msg_recp, _msg_recv, _features in zip(state[1], msg_recp, msg_recv, features)])
            
            roll.msgs += [[_msg.data.numpy() for _msg in msg]]
            roll.msg_recps += [[_msg_recp.data.numpy() for _msg_recp in msg_recp]]
            roll.features += [[[_f.data.numpy() for _f in _feature] for _feature in features]]
            roll.features_omni += [[_feature.data.numpy() for _feature in features_omni]]

            #print(logit)
            #action = [F.softmax(_logit).multinomial().data.numpy() for _logit in logit]
            action = [F.softmax(_logit).clamp(max=1 - 1e-20).multinomial().data.numpy() for _logit in logit]

            #'''#TODO: only agent at specified index performs non-linguistic actions for this; need to undo later
            action = action[single_actor_settings(args)]
            #'''

            if action_init and (episode_length < len(action_init)):
                state, reward, done, info = env.step([action_init[episode_length]])
            else:
                state, reward, done, _ = env.step(action[0])

            #env.render()

            done = done[0] or episode_length >= args.max_episode_length or state_stuck(args.env_name, state)
            reward = sum([max(min(_reward, 1), -1) for _reward in reward])
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

                log_value('train'+str(rank)+'/episode_length', episode_length)
                log_value('train'+str(rank)+'/reward_sum', reward_sum)
                episode_length = 0
                reward_sum = 0

            roll.actions += [action]
            roll.rewards += [reward]
            roll.terminal = done
            
            if done:
                break

        if len(roll.rewards) > 0:
            # hack to get agent to learn to survive
            roll.rewards[-1] += args.survival_reward
            reward_sum += args.survival_reward

            if not action_init:
                if args.throughput_log:
                    fwd_count += len(roll.rewards)
                yield roll
            else:
                if episode_length > len(action_init):
                    if args.throughput_log:
                        fwd_count += len(roll.rewards)
                    yield roll

def fwd(rank, args, shared_model, fwd_q, fwd_count, **kwargs):
    from env_util.envs import create_env
    torch.manual_seed(args.seed + rank)

    env = create_env(args.env_name, rank, 1, **kwargs)
    env.seed([args.seed + rank])

    ob_space__all = [ob.shape for ob in env.observation_space]

    #HACK
    action_space__all = [env.action_space.n for _ in range(args.num_agents)]

    #model = Comm(ob_space__all, action_space__all, env.action_space.n, args)
    model = Comm(env.observation_space_omni.shape, ob_space__all, action_space__all, env.action_space.n, args)
    model.train()

    rollout_provider = env_runner(rank, args, env, model, shared_model, fwd_count)
    while True:

        # the timeout variable exists because apparently, if one worker dies, the other workers
        # won't die with it, unless the timeout is set to some large number.  This is an empirical
        # observation.
        fwd_q.put(next(rollout_provider), timeout=600.0)

def replay_server(args, replay_buffer, fwd_q, bwd_q, fwd_count, bwd_count):
    onp_cache = deque(maxlen=6*args.num_processes)
    #onp_cache = deque(maxlen=1)
    last_save_time = time.time()
    last_fwd_count = Variable(fwd_count.data.clone(), requires_grad=False)
    last_bwd_count = Variable(bwd_count.data.clone(), requires_grad=False)
    while True:
        rolls = []
        if not args.offp:
            rolls.append(fwd_q.get(timeout=600.0))
        else:
            try:
                rolls.append(fwd_q.get(timeout=5e-4))
            except:
                pass

        while not fwd_q.empty():
            try:
                rolls.append(fwd_q.get(timeout=1e-4))
            except:
                break

        if args.throughput_log:
            if (time.time() - last_save_time) > 60:     
                last_save_time = time.time()
                log_value('train/fwd_steps_per_minute', (fwd_count-last_fwd_count).data.numpy()[0])
                log_value('train/bwd_steps_per_minute', (bwd_count-last_bwd_count).data.numpy()[0])
                log_value('train/bwd_2_fwd_ratio', (bwd_count-last_bwd_count).data.numpy()[0]/(fwd_count+1-last_fwd_count).data.numpy()[0])
                log_value('train/fwd_qsize', fwd_q.qsize())
                log_value('train/bwd_qsize', bwd_q.qsize())
                log_value('train/onp_cache_size', len(onp_cache))
                last_fwd_count = Variable(fwd_count.data.clone(), requires_grad=False)
                last_bwd_count = Variable(bwd_count.data.clone(), requires_grad=False)

        #TODO: maybe have replay buffer save to file like chainerrl does so that it doesn't eat up ram'''

        if args.offp:
            for roll in rolls:
                replay_buffer.add(roll)

        rolls_placed = 1
        if args.onp:
            rolls_placed = 0
            if not bwd_q.full():
                try:
                    for rdx, roll in enumerate(rolls):
                        bwd_q.put(roll, timeout=1e-4)
                        rolls_placed += 1
                except:
                    if args.offp:
                        for roll in rolls[rdx:]:
                            onp_cache.append(roll)
            else:
                if args.onp_cache:
                    for roll in rolls:
                        onp_cache.append(roll)

            if not bwd_q.full() and args.onp_cache:
                try:
                    for _ in range(len(onp_cache)):
                        last_pop = onp_cache.pop()
                        bwd_q.put(last_pop, timeout=1e-4)
                except:
                    onp_cache.append(last_pop)

        # Only goes to off-policy if all on_policy rollouts have already been used
        if args.offp:
            if not bwd_q.full():
                if len(onp_cache) == 0:
                    for _ in range(max(args.off_policy_rate, rolls_placed*args.off_policy_rate)):
                        if replay_buffer.trainable() and not bwd_q.full():
                            try:
                                bwd_q.put(replay_buffer.sample(), timeout=1e-4)
                            except:
                                pass

def bwd(rank, args, shared_model, shared_model_avg, shared_discrim_model, bwd_q, env_details, dirichlet_vocab, bwd_count, d_acc_shared, optimizer=None, d_optimizer=None):
    torch.manual_seed(args.seed + rank)

    ob_space__omni, ob_space__all, action_space__all = env_details[0], env_details[1], env_details[2]

    #model = Comm(ob_space__all, action_space__all, action_space__all[0], args)
    model = Comm(ob_space__omni, ob_space__all, action_space__all, action_space__all[0], args)
    if args.trpo:
        model_avg = Comm(ob_space__all, action_space__all, action_space__all[0], args)
        model_avg.train()
    model.train()

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)

    if args.trpo:
        alpha = torch.FloatTensor([args.alpha])

    if args.pf:
        from util.prof_force.discrim import Discrim
        discrim_model = Discrim(args)
        discrim_model.train()
        d_acc_avg_proportion = 1/args.d_acc_avg_size
        if d_optimizer is None:
            d_optimizer = optim.Adam(shared_discrim_model.parameters(), lr=args.lr)

    while True:
        roll = bwd_q.get(timeout=600.0)

        torch.manual_seed(args.seed + roll.rank)

        model.load_state_dict(shared_model.state_dict())
        if args.trpo:
            model_avg.load_state_dict(shared_model_avg.state_dict())
        if args.pf:
            discrim_model.load_state_dict(shared_discrim_model.state_dict())

        states, actions, rewards, probs, values, R, features, features_omni, msgs, msg_recps, done = \
            roll.states, roll.actions, roll.rewards, roll.probs, roll.values, roll.R, roll.features, roll.features_omni, roll.msgs, roll.msg_recps, roll.terminal

        # init with first hidden states from fwd process's rollout
        feature = feature_avg =[[Variable(torch.from_numpy(_f)) for _f in _feature] for _feature in features[0]]
        feature_omni = feature_omni_avg =[Variable(torch.from_numpy(_feature)) for _feature in features_omni[0]]
        msg_recp = msg_recp_avg = [Variable(torch.from_numpy(_msg_recp)) for _msg_recp in msg_recps[0]]
        msg = msg_avg = [Variable(torch.from_numpy(_msg)) for _msg in msgs[0]]

        values = []
        log_probs = []
        entropies = []

        if args.trpo:
            probs_avg = []
            log_probs_dist = []

        if args.pf:
            features_forced = []
            features_omni_forced = []
            msg_recps_forced = []
            msgs_forced = []

        for sdx, state in enumerate(states):
            states[sdx] = env_from_numpy(states[sdx])

        for s in range(len(rewards)):
            msg_recv = swap_msgs(msg, msg_recp)
            value, logit, msg_recp, msg, feature, feature_omni = model(Variable(states[s][0].float().unsqueeze(0)), feature_omni, [(Variable(_state.float().unsqueeze(0)), _msg_recp, _msg_recv, _feature) \
                        for _state, _msg_recp, _msg_recv, _feature in zip(states[s][1], msg_recp, msg_recv, feature)])
            if args.pf:
                features_forced+=[feature]
                features_omni_forced+=[feature_omni]
                msg_recps_forced+=[msg_recp]
                msgs_forced+=[msg]

            #'''
            if args.dirichlet_vocab:
                #print(dirichlet_vocab.reward(msg))
                rewards[s] += (dirichlet_vocab.reward(msg)*args.d_weight)/len(rewards)
                dirichlet_vocab.update(msg)
            #'''

            prob = [F.softmax(_logit).clamp(max=1 - 1e-20) for _logit in logit]
            log_prob_dist = [_prob.log() for _prob in prob]

            #TODO: multiple nonlingustic actions
            entropy = -(log_prob_dist[single_actor_settings(args)] * prob[single_actor_settings(args)]).sum(1)
            entropies+=[entropy]

            if args.trpo:
                log_probs_dist.append(log_prob_dist)

                #msg_recv_avg = swap_msgs(msg_avg)
                msg_recv_avg = swap_msgs(msg_avg, msg_recp_avg)
                _, logit_avg, msg_recp_avg, msg_avg, feature_avg, feature_omni_avg = model_avg(Variable(states[s][0].float().unsqueeze(0)), feature_omni_avg, [(Variable(_state.float().unsqueeze(0)), _msg_recp, _msg_recv, _feature) \
                    for _state, _msg_recp, _msg_recv, _feature in zip(states[s][1], msg_recp_avg, msg_recv_avg, feature_avg)])
                
                prob_avg = [F.softmax(_logit_avg) for _logit_avg in logit_avg]
                probs_avg.append(prob_avg)

            '''#TODO: only 0th agent performs non-linguistic actions for this; need to undo later'''
            log_prob = [_log_prob_dist.gather(1, Variable(torch.from_numpy(actions[s]))) for _log_prob_dist in log_prob_dist]
            #log_prob = [_log_prob_dist.gather(1, Variable(torch.from_numpy(_action))) for _logit, _action in zip(logit, actions[s])]

            values+=[value]
            log_probs+=[log_prob]

        R = [torch.zeros(1, 1) for _ in range(args.num_agents)]
        if not done:
            msg_recv = swap_msgs(msg, msg_recp)
            value, _, _, _, _, _ = model(Variable(states[-1][0].float().unsqueeze(0)), feature_omni, [(Variable(_state.float().unsqueeze(0)), _msg_recp, _msg_recv, _feature) \
                        for _state, _msg_recp, _msg_recv, _feature in zip(states[-1][1], msg_recp, msg_recv, feature)])
            R = [_value.data for _value in value]

        R = [Variable(_R) for _R in R]

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

        if args.pf:
            '''TODO: ADD features_omni TO pf loss'''
            if len(features) > 2: # sometimes length is =<2 at done cutoff
                free__tensor = Variable(torch.stack([torch.cat(n_list_to_t_list(_f)+n_list_to_t_list(_m)+n_list_to_t_list(_mr), 1) for _f, _m, _mr in zip(features[1:], msgs[1:], msg_recps[1:])], 0))
                forced__tensor = torch.stack([torch.cat(_f+_m+_mr, 1) for _f, _m, _mr in zip(features_forced, msgs_forced, msg_recps_forced)], 0)
                d_loss, gd_loss, d_acc = discrim_loss(args, discrim_model, free__tensor.permute(1,0,2), forced__tensor.permute(1,0,2))
            else:
                d_loss, gd_loss, d_acc = Variable(torch.Tensor([0])), Variable(torch.Tensor([0])), Variable(torch.Tensor([-1]))

            if d_acc.data.numpy()[0]!=-1:
                d_acc_shared = (d_acc_shared + d_acc_avg_proportion*d_acc)/(1+d_acc_avg_proportion)

        # update g based on pcl loss
        optimizer.zero_grad()
        if args.pf:
            loss.backward(retain_graph=True)
        else:
            loss.backward()
        rl_norm = torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        ensure_shared_grads(model, shared_model)
        optimizer.step()

        if args.pf:
            log_value('train/g_loss', loss.data.numpy()[0])
            if d_loss.data.numpy()[0]!=0:
                log_value('train/d_acc_shared', d_acc_shared.data.numpy()[0])
                log_value('train/d_loss', d_loss.data.numpy()[0])
                log_value('train/gd_loss', gd_loss.data.numpy()[0])
                if d_acc_shared.data.numpy()[0] > .75:
                    optimizer.zero_grad()
                    (args.pf_weight * gd_loss).backward(retain_graph=True)
                    g_pf_norm = torch.nn.utils.clip_grad_norm(model.parameters(), min(rl_norm, 40))
                    log_value('train/clipped_rl_norm_over_clipped_g_pf_norm', min(rl_norm, 40)/min(g_pf_norm, 40))
                    ensure_shared_grads(model, shared_model)
                    optimizer.step()
                if d_acc_shared.data.numpy()[0] < .99:
                    d_optimizer.zero_grad()
                    d_loss.backward()
                    d_norm = torch.nn.utils.clip_grad_norm(discrim_model.parameters(), 40)
                    log_value('train/d_norm', d_norm)
                    ensure_shared_grads(discrim_model, shared_discrim_model)
                    d_optimizer.step()

        if args.trpo:
            update_avg(shared_model_avg, model, alpha)

        if args.throughput_log:
            bwd_count += len(rewards)
