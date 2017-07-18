import numpy as np
import os
import torch
from torch.autograd import Variable

from tensorboard_logger import configure, log_value

def load_vocab(vocab_name, max_vocab_size=-1): 
    '''TODO: FIX VOCAB ZERO PAD AND LENGTH if the utterance_length (and/or whether or not to utter) is not static'''
    if vocab_name == "basic-english":
        SCRIPT_PATH = os.path.dirname(__file__)
        print(SCRIPT_PATH)
        with open(SCRIPT_PATH + "/" + "basic_english_vocab.txt", "r") as words_f:
            #WORDS = ["_PAD"] + [line.strip() for line in words_f]
            WORDS = [line.strip() for line in words_f]
    else:
        pass

    if max_vocab_size != -1:
        WORDS = WORDS[0:max_vocab_size]
    else:
        pass   
    return WORDS

def multi_obs_settings(args):
    if args.env_name == "Catcher-v0":
        _multi_obs_settings = (.90, .03, True)
    elif args.env_name == "BreakoutDeterministic-v4":
        _multi_obs_settings = (.93, .03, True)
    elif args.env_name == "PongDeterministic-v4":
        #Pong is for debug; not usable for comm because it cheats by staying in 1 place
        #_multi_obs_settings = (.90, .03, True)
        _multi_obs_settings = (.60, .19, True)
    elif args.env_name == "flashgames.RetroRunner-v0":
        _multi_obs_settings = (.21, .07, False)
    elif args.env_name == "flashgames.RunRamRun-v0":
        _multi_obs_settings = (.21, .07, False)
    else:
        raise ValueError("multi_obs_settings for this env not defined")
    return _multi_obs_settings

def env_from_numpy(state):
    state = [torch.from_numpy(state[0]), [torch.from_numpy(_state) for _state in state[1]]]
    return state

def single_actor_settings(args):
    if args.env_name == "Catcher-v0":
        acting_agent_idx = 1
    elif args.env_name == "BreakoutDeterministic-v4":
        acting_agent_idx = 1
    elif args.env_name == "PongDeterministic-v4":
        #Pong is for debug; not usable for comm because it cheats by staying in 1 place
        acting_agent_idx = 1
    elif args.env_name == "flashgames.RetroRunner-v0":
        acting_agent_idx = 0
    elif args.env_name == "flashgames.RunRamRun-v0":
        acting_agent_idx = 0
    else:
        raise ValueError("single_actor_settings for this env not defined")
    return acting_agent_idx

def swap_msgs(msg_sent, msg_recp):
    def index_except(_a, _i):
        return [__a[_i] for __a in _a]

    # message is only received by recipients
    def recp_gate(_msg_recv, _msg_recp):
        if len(_msg_recv) > 1:

            return [__msg_recv*__msg_recp.expand_as(__msg_recv) for __msg_recv, __msg_recp in zip(_msg_recv, _msg_recp)]
        else:
            return _msg_recv

    return [[msg_sent[idx], recp_gate(msg_sent[:idx]+msg_sent[idx+1:], rp)] for idx, (s, rp) in enumerate(zip(msg_sent, msg_recp))]

    #return [index_except(__msg_sent[:idx], idx-1) + index_except(__msg_sent[idx+1:], idx) for idx, i in enumerate(__msg_sent)]
    # ^version with owns previous msg

def utter_penalty(args):
    if msg != 0:
        reward -= args.utter_penalty

class DirichletVocab(object):
    def __init__(self, args):
        self.args = args

        self.n = Variable(torch.FloatTensor([2]).share_memory_(), requires_grad=False)
        self.n_ks = [Variable(torch.FloatTensor(1).share_memory_().zero_(), requires_grad=False) for v in range(args.max_vocab_size-1)]

        #TODO: load from previous runs
        if args.load:
            pass

    def update(self, msgs):
        for msg in msgs:
            self.n_ks[torch.max(msg, 1)[1].data.numpy()[0]-1] += 1
            self.n+=1
        #log_value()

    def reward(self, msgs):
        r=0
        '''TODO: switch back to log prob because it scales better when summing'''
        for i, msg in enumerate(msgs):
            # 0 idx represents silence; it still has an embedding because silence actually has semantic content
            # slight penalty for non-silence
            if torch.max(msg, 1)[1].data.numpy()[0] != 0:
                '''TODO: maybe divide bu utter penalty instead of subtracting utter_penalty'''
                r+=float((self.n_ks[torch.max(msg, 1)[1].data.numpy()[0]-1]/(self.args.d_alpha+self.n-1)).data.numpy()[0])/self.args.utter_penalty
            else:
                r+=float((torch.max(torch.stack(self.n_ks, 0).squeeze(), 0)[0]/(self.args.d_alpha+self.n-1)).data.numpy()[0])
        r /= len(msgs)

        return r

''' # Lines below are used when env split/mask is not done through gym/uni api'''
''' # I'm doing env splitting/masking through gym api, so lines below are currently not being used.'''
def env_splitter(env_step_observation, split_ratio, split_hor_line, num_of_agents):
    """
Splits the observation of the environment (with a horizontal line), such that 
the observer sees fraction of environment equal to split_ratio and actor can only
see the remainder. This split forces the observer and actor to learn to communicate 
in order to complete task. Can be applied to almost any arbitrary environment.
Is asymmetric such that the observer directly observes much more of environment,
but the actor has all/more_of the controls (e.g. control of the paddle in pong).
"""
    """TODO: add argument to choose whether split is horizontal or vertical"""
    """Does gym/univers have a wrapper/util for splitting?"""
    """add/test out pong as well"""
    """split into variable num for variable agents"""
    #split_point = int(split_ratio * env_step_observation.shape[0])
    observer_obs = env_step_observation[0:split_point].tolist()
    actor_obs = env_step_observation[split_point:env_step_observation.shape[0]].tolist()

    #:,0:split_point
    #:,split_point:env_step_observation.shape[0]

    return actor_obs, observer_obs

def env_multi_splitter(env_step_observation, split_ratio1, split_ratio2, split_hor_line, num_of_agents, channel_first=True):
    """
Splits the observation of the environment (with a horizontal line), such that 
the observer sees fraction of environment equal to split_ratio and actor can only
see the remainder. This split forces the observer and actor to learn to communicate 
in order to complete task. Can be applied to almost any arbitrary environment.
Is asymmetric such that the observer directly observes much more of environment,
but the actor has all/more_of the controls (e.g. control of the paddle in pong).
"""
    """TODO: add argument to choose whether split is horizontal or vertical"""
    """Does gym/universe have a wrapper/util for splitting?"""
    """add/test out pong as well"""
    """split into variable num for variable agents"""

    #split_point = int(split_ratio * env_step_observation.shape[0])

    #print(split_point1)
    #print(split_point2)

    if channel_first==True:
        split_point1 = int(split_ratio1 * env_step_observation.shape[1])
        split_point2 = int(split_ratio2 * env_step_observation.shape[1])
        if split_hor_line == True:
            '''
            observer_obs = env_step_observation[:,0:split_point1].tolist()
            actor_obs = env_step_observation[:,split_point2:env_step_observation.shape[1]].tolist()
            '''

            observer_obs = env_step_observation[:,0:split_point1]
            actor_obs = env_step_observation[:,split_point2:env_step_observation.shape[1]]
        else:
            '''
            observer_obs = env_step_observation[:,:,0:split_point1].tolist()
            actor_obs = env_step_observation[:,:,split_point2:env_step_observation.shape[1]].tolist()
            '''

            observer_obs = env_step_observation[:,:,0:split_point1]
            actor_obs = env_step_observation[:,:,split_point2:env_step_observation.shape[1]]
    else:
        split_point1 = int(split_ratio1 * env_step_observation.shape[0])
        split_point2 = int(split_ratio2 * env_step_observation.shape[0])
        if split_hor_line == True:
            observer_obs = env_step_observation[0:split_point1].tolist()
            actor_obs = env_step_observation[split_point2:env_step_observation.shape[0]].tolist()
        else:
            observer_obs = env_step_observation[:,0:split_point1].tolist()
            actor_obs = env_step_observation[:,split_point2:env_step_observation.shape[0]].tolist()

    '''
    print("actor_obs")
    print(np.asarray(actor_obs).shape)

    print("observer_obs")
    print(np.asarray(observer_obs).shape) 
    #'''  

    return actor_obs, observer_obs

def env_masker(env_step_observation, split_ratio, split_hor_line, num_of_agents):
    """
Splits the observation of the environment (with a horizontal line), such that 
the observer sees fraction of environment equal to split_ratio and actor can only
see the remainder. This split forces the observer and actor to learn to communicate 
in order to complete task. Can be applied to almost any arbitrary environment.
Is asymmetric such that the observer directly observes much more of environment,
but the actor has all/more_of the controls (e.g. control of the paddle in pong).
"""
    """TODO: add argument to choose whether split is horizontal or vertical"""
    """Does gym/univers have a wrapper/util for splitting?"""
    """add/test out pong as well"""
    """split into variable num for variable agents"""
    split_point = int(split_ratio * env_step_observation.shape[0])
    #print("env_step_observation.shape")
    #print(env_step_observation.shape)
    if split_hor_line == True:
        observer_obs = np.concatenate((env_step_observation[0:split_point], np.zeros_like(env_step_observation[split_point:env_step_observation.shape[0]]))).tolist()
        actor_obs = np.concatenate((np.zeros_like(env_step_observation[0:split_point]), env_step_observation[split_point:env_step_observation.shape[0]])).tolist()
    else:
        observer_obs = np.concatenate((env_step_observation[:,0:split_point], np.zeros_like(env_step_observation[:,split_point:env_step_observation.shape[0]])), axis=1).tolist()
        actor_obs = np.concatenate((np.zeros_like(env_step_observation[:,0:split_point]), env_step_observation[:,split_point:env_step_observation.shape[0]]), axis=1).tolist()

    return actor_obs, observer_obs

def env_multi_masker(env_step_observation, split_ratio1, split_ratio2, split_hor_line, num_of_agents):
    """
Splits the observation of the environment (with a line), such that 
the observer sees fraction of environment equal to split_ratio1 and actor can only
see the remainder of split_ratio2. This split forces the observer and actor to learn to communicate 
in order to complete task. Can be applied to almost any arbitrary environment.
Is asymmetric such that the observer directly observes much more of environment,
but the actor has all/more_of the controls (e.g. control of the paddle in pong).
"""
    """TODO: add argument to choose whether split is horizontal or vertical"""
    """Does gym/univers have a wrapper/util for splitting?"""
    """add/test out pong as well"""
    """split into variable num for variable agents"""
    split_point1 = int(split_ratio1 * env_step_observation.shape[0])
    split_point2 = int(split_ratio2 * env_step_observation.shape[0])
    #print("env_step_observation.shape")
    #print(env_step_observation.shape)
    if split_hor_line == True:
        observer_obs = np.concatenate((env_step_observation[0:split_point1], np.zeros_like(env_step_observation[split_point1:env_step_observation.shape[0]]))).tolist()
        actor_obs = np.concatenate((np.zeros_like(env_step_observation[0:split_point2]), env_step_observation[split_point2:env_step_observation.shape[0]])).tolist()
    else:
        observer_obs = np.concatenate((env_step_observation[:,0:split_point1], np.zeros_like(env_step_observation[:,split_point1:env_step_observation.shape[0]])), axis=1).tolist()
        actor_obs = np.concatenate((np.zeros_like(env_step_observation[:,0:split_point2]), env_step_observation[:,split_point2:env_step_observation.shape[0]]), axis=1).tolist()

    return actor_obs, observer_obs

def env_custom(env_step_observation, *args):
    crop_point = int(.5 * env_step_observation.shape[0])

    split_ratio1, split_ratio2, split_hor_line, num_of_agents = args
    split_point1 = int(split_ratio1 * env_step_observation.shape[0])
    split_point2 = int(split_ratio2 * env_step_observation.shape[0])

    observer_obs = np.concatenate((env_step_observation[0:split_point1], np.zeros_like(env_step_observation[split_point1:env_step_observation.shape[0]]))).tolist()
    actor_obs = np.concatenate((np.zeros_like(env_step_observation[crop_point:split_point2]), env_step_observation[split_point2:env_step_observation.shape[0]])).tolist()

    return actor_obs, observer_obs

def env_dummy(env_step_observation, *args):
    return [env_step_observation for i in range(2)]

'''
class CommSpec(object):
    def __init__(self, num_of_agents, env_id, vocab, utterance_length, temperature, comm_allowed=True):
        self.comm_allowed = comm_allowed
        self.num_of_agents = num_of_agents
        self.env_id = env_id
        self.vocab = vocab
        self.vocab_space = len(self.vocab)
        self.utterance_length = utterance_length
        self.temperature = temperature

    def update_temp(self, temperature):
        self.temperature = temperature

class agent_inputs__all(object):
    def __init__(self, obs, msg, features):
        self.obs = obs
        self.msgs = msgs
        self.features = features

class agent_inputs(object):
    def __init__(self, ob, msg, features):
        self.ob = ob
        self.msg = msg
        self.features = features
        #'''

class CommSpec(object):
    def __init__(self, args):
        self.args = args

    def multi_obs_settings():
        if self.args.env_name == "Catcher-v0":
            _multi_obs_settings = (.90, .03, True)
        elif self.args.env_name == "flashgames.RetroRunner-v0":
            _multi_obs_settings = (.21, .07, False)
        else:
            raise ValueError("multi_obs_settings for this env not defined")
        return _multi_obs_settings

    def single_actor_settings():
        if self.args.env_name == "Catcher-v0":
            acting_agent_idx = 1
        elif self.args.env_name == "flashgames.RetroRunner-v0":
            acting_agent_idx = 0
        else:
            raise ValueError("single_actor_settings for this env not defined")
        return acting_agent_idx
