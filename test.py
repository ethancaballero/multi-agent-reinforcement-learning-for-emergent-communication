import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Comm
from torch.autograd import Variable
from torchvision import datasets, transforms
import time
from collections import deque

from util.comm_util import *

import numpy as np
from env_util.env_fixes import init_actions, state_stuck

from tensorboard_logger import configure, log_value


def test(rank, args, shared_model, dirichlet_vocab, **kwargs):
    from env_util.envs import create_env
    torch.manual_seed(args.seed + rank)

    env = create_env(args.env_name, rank, 1, **kwargs)
    env.seed([args.seed + rank])

    ob_space__all = [ob.shape for ob in env.observation_space]

    #HACK
    action_space__all = [env.action_space.n for _ in range(args.num_agents)]

    model = Comm(env.observation_space_omni.shape, ob_space__all, action_space__all, env.action_space.n, args)
    model.eval()

    state = env_from_numpy(env.reset())
    reward_sum = 0
    done = True

    dirichlet_reward = 0
    word_used_bools = [0 for _ in range(args.max_vocab_size)]

    start_time = last_save_time = time.time()

    actions = deque(maxlen=1000)
    episode_length = 0
    #highest_reward=0
    action_init = init_actions(args.env_name)
    timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
    num_tests=0

    if args.demonstrate:
        print('screen is split: ', str(multi_obs_settings(args)))
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            msg = [Variable(torch.zeros(1, args.max_vocab_size), volatile=True) for _ in range(args.num_agents)]
            msg_recp = [Variable(torch.zeros(1, args.num_agents-1), volatile=True) for _ in range(args.num_agents)]
            features = [(Variable(torch.zeros(1, args.hidden_size), volatile=True), Variable(torch.zeros(1, args.hidden_size), volatile=True)) for _ in range(args.num_agents)]
            features_omni = [Variable(torch.zeros(1, args.hidden_size), volatile=True), Variable(torch.zeros(1, args.hidden_size), volatile=True)]
        else:
            msg = [Variable(_msg.data, volatile=True) for _msg in msg]
            msg_recp = [Variable(_msg_recp.data, volatile=True) for _msg_recp in msg_recp]
            features = [(Variable(_f.data, volatile=True) for _f in _feature) for _feature in features]
            features_omni = [Variable(torch.zeros(1, args.hidden_size), volatile=True), Variable(torch.zeros(1, args.hidden_size), volatile=True)]

        msg_recv = swap_msgs(msg, msg_recp)
        _, logit, msg_recp, msg, features, _ = model(Variable(state[0].float().unsqueeze(0), volatile=True), features_omni, [(Variable(_state.float().unsqueeze(0), volatile=True), _msg_recp, _msg_recv, _features) \
                    for _state, _msg_recp, _msg_recv, _features in zip(state[1], msg_recp, msg_recv, features)])

        if args.dirichlet_vocab:
            dirichlet_reward+=dirichlet_vocab.reward(msg)
        for mdx, _msg in enumerate(msg):
            '''TODO: add msg_recp gate'''
            msg_index = int(torch.max(_msg, 1)[1].data.numpy()[0])
            for recp in range(args.num_agents-1):
                if recp == mdx:
                    recp += 1
                else: 
                    pass
                if not num_tests % 20:
                    log_value('test/msg_all', msg_index)
                    log_value('test/msg_sent_from_'+str(mdx)+'_to_'+str(recp), msg_index)
                if args.demonstrate:
                    if recp == single_actor_settings(args):
                        print('msg_sent_from_'+str(mdx)+'_to_'+str(recp)+': ', msg_index)
                word_used_bools[msg_index] = 1

        prob = [F.softmax(_logit) for _logit in logit]
        action = [_prob.max(1)[1].data.numpy() for _prob in prob]

        #'''#TODO: only agent at specified index performs non-linguistic actions for this; need to undo later
        action = action[single_actor_settings(args)]
        #'''

        if action_init and (episode_length < len(action_init)):
            state, reward, done, info = env.step([action_init[episode_length]])
        else:
            state, reward, done, info = env.step(action)

        if args.demonstrate:
            env.render()

        done = done[0] or episode_length >= args.max_episode_length or state_stuck(args.env_name, state)
        reward_sum += sum(reward)

        if args.debug_vision:
            import scipy.misc
            for _sdx, _state in enumerate(state):
                scipy.misc.toimage(_state[0], cmin=0.0, cmax=1.0, channel_axis=0).save('tmp/'+str(episode_length)+'.'+str(_sdx)+'.jpg')
                scipy.misc.toimage(_state[0], cmin=0.0, cmax=1.0, channel_axis=0).save('tmp/'+str(episode_length)+'.'+str(_sdx)+'.jpg')

        if info:
            for k, v in info.items():
                if k=='n': 
                    if len(v[0]) > 0:
                        for _k in zip(v[0].keys()):
                            log_value(str(_k[0]), v[0][str(_k[0])])
                else:
                    log_value(k, v)

        if args.train_pause:
            actions.append(action[0])
            if actions.count(actions[0]) == actions.maxlen:
                done = True

        if done:
            log_value('test/reward_sum', reward_sum)
            log_value('test/episode_length', episode_length)
            log_value('test/dirichlet_reward', dirichlet_reward/episode_length)
            log_value('test/unique_words_used', sum(word_used_bools))

            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))
            reward_sum = 0
            episode_length = 0
            num_tests+=1
            dirichlet_reward = 0
            word_used_bools = [0 for _ in range(args.max_vocab_size)]

            actions.clear()

            if args.save == True:
                if (time.time() - last_save_time) > args.save_interval:
                    last_save_time = time.time()
                    torch.save(model.state_dict(), './weights/'+args.save_file+'.pth')

            if args.test_sleep:
                time.sleep(60)

            if args.demonstrate:
                print('screen is split: ', str(multi_obs_settings(args)))

            state = env.reset()

        state = env_from_numpy(state)
