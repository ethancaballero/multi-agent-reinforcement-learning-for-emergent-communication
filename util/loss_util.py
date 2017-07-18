import torch
from torch.autograd import Variable
import numpy as np
import random
from torch.nn import functional as F
import torch.nn as nn

import util.comm_util as comm_util

def pcl_loss(args, rewards, values, log_probs, R):
    #TODO: Make version in which agents don't share reward

    next_values = values[1:] + [R]
    
    pi_loss=0
    v_loss=0
    seq_len=len(rewards)

    #add randomness to rollout_len to promote path consistency for multiple lengths'''
    #rollout_len = args.rollout_len + random.randrange(0,41,20)

    #for t in range(t_start, t_stop):
    for a in range(args.num_agents):
        if a==comm_util.single_actor_settings(args):
            for t in range(len(rewards)):
                d = min(len(rewards) - t, args.rollout_len)
                # Discounted sum of immediate rewards
                R_seq = sum(args.gamma ** i * rewards[t + i] for i in range(d))
                # Discounted sum of log likelihoods
                G = sum(log_probs[t + i][a] * args.gamma ** i for i in range(d))

                #G = G.unsqueeze(-1)
                G = G.unsqueeze(len(G.size()))
                last_v = next_values[t + d - 1][a]

                '''
                if not args.backprop_future_values:
                    last_v = Variable(last_v.data)
                    #'''

                # C_pi only backprop through pi
                C_pi = (- values[t][a].detach() +
                        args.gamma ** d * last_v.detach() +
                        R_seq -
                        args.tau * G)

                # C_v only backprop through v
                C_v = (- values[t][a] +
                       args.gamma ** d * last_v +
                       R_seq -
                       args.tau * G.detach())

                pi_loss += C_pi ** 2
                v_loss += C_v ** 2

    pi_loss /= 2
    v_loss /= 2

    # Re-scale pi loss so that it is independent from tau
    pi_loss /= args.tau

    pi_loss *= args.pi_loss_coef
    v_loss *= args.v_loss_coef

    if args.normalize_loss_by_steps:
        pi_loss /= seq_len
        v_loss /= seq_len

    return pi_loss, v_loss

def a3c_loss(args, rewards, values, log_probs, entropies, R):
    #TODO: Make version in which agents don't share reward
    values.append(R)
    policy_loss = 0
    value_loss = 0
    gae = torch.zeros(1, 1) 
    for a in range(args.num_agents):
        if a==comm_util.single_actor_settings(args):
            R = R[a]
            for i in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[i]
                advantage = R - values[i][a]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                # Generalized Advantage Estimataion
                delta_t = rewards[i] + args.gamma * \
                    values[i + 1][a].data - values[i][a].data
                gae = gae * args.gamma * args.tau + delta_t

                policy_loss = policy_loss - \
                    log_probs[i][a] * Variable(gae) - 0.01 * entropies[i]

    return policy_loss*args.pi_loss_coef, value_loss*args.v_loss_coef

def kl_div(avg_probs, current_log_probs, args):
    '''TODO: remove single_actor hack'''
    a=comm_util.single_actor_settings(args)

    total_kl=0
    for avg_prob, current_log_prob in zip(avg_probs, current_log_probs):
        total_kl+= F.kl_div(current_log_prob[a], avg_prob[a], size_average=False)
    return total_kl

def loss_with_kl_constraint(avg_probs, current_log_probs,
                            model, original_loss, delta, optimizer, args):
    """Compute loss considering a KL constraint.

    Args:
        current is from most recent (current) policy
        avg is from shared avg policy
        original_loss : Loss to minimize
        delta (float): 
    Returns:
        loss 
    """

    # Compute g: a direction to minimize the original loss
    optimizer.zero_grad()
    original_loss.backward(retain_graph=True)
    g = [Variable(p.grad.data.clone()) if p.grad is not None else Variable(torch.zeros(1)) for p in model.parameters()]

    # Compute k: a direction to increase KL div.
    optimizer.zero_grad()
    kl = kl_div(avg_probs, current_log_probs, args)
    (-kl).backward(retain_graph=True)
    k = [Variable(p.grad.data.clone()) if p.grad is not None else Variable(torch.zeros(1)) for p in model.parameters()]

    optimizer.zero_grad()

    kg_dot = sum(torch.sum(k_p * g_p) for k_p, g_p in zip(k, g))
    kk_dot = sum(torch.sum(k_p ** 2) for k_p in k)

    if kk_dot.data.numpy()[0] > 0:
        k_factor = torch.clamp((kg_dot-delta)/kk_dot, min=0)
    else:
        k_factor = Variable(torch.zeros(1))

    z = [gp - k_factor.expand_as(kp) * kp for kp, gp in zip(k, g)]
    loss = 0

    for p, zp in zip(model.parameters(), z):
        if p.size() == zp.size():
            loss += torch.sum(p * zp)
    return loss, kl.data

#TODO: Better GANs
def discrim_loss(args, d_model, features_free, features_forced):
    criterion = nn.BCELoss()
    #print(features_free)
    free_guess = d_model(features_free)
    forced_guess = d_model(features_forced)

    d_free_error = criterion(free_guess, Variable(torch.ones(1)))  # ones = true
    d_fake_error = criterion(forced_guess.detach(), Variable(torch.zeros(1)))  # zeros = fake

    # makes forced look like free
    gd_forced_error = criterion(forced_guess, Variable(torch.ones(1)))  # we want to fool, so pretend it's all free # makes forced look like free

    # makes free look like forced # paper uses both
    gd_free_error = 0
    #gd_free_error = criterion(free_guess, Variable(torch.zeros(1)))  # we want to fool, so pretend it's all forced  # makes free look like forced

    d_acc = (torch.round(free_guess) + torch.round(forced_guess)) / 2
    return (d_free_error + d_fake_error, gd_forced_error + gd_free_error, d_acc[0].detach())
