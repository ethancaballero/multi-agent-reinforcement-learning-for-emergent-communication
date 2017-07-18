import numpy as np

'''TODO: save to file like chainerrl does so that it doesn't eat up ram'''
class ReplayBuffer(object):
    #def __init__(self, max_len=100000, alpha=1):
    def __init__(self, args):
        self.args = args
        self.max_len = args.replay_buffer_size
        self.alpha = 0.6
        self.buffer = []
        # weight is not normalized
        '''normalize weight for varying rollout lengths via maybe /exp^-len(rewards)'''
        self.weight = np.array([])

    def add(self, episode):
        self.buffer.append(episode)
        #self.weight = np.append(self.weight, np.exp(self.alpha*episode['rewards'].sum()))
        if self.args.death_penalty > 0:
            self.weight = np.append(self.weight, np.exp(self.alpha*np.absolute(np.asarray(episode.rewards).sum())))
        else:
            self.weight = np.append(self.weight, np.exp(self.alpha*np.asarray(episode.rewards).sum()))
        #self.weight = np.append(self.weight, np.exp(self.alpha*np.asarray(episode.rewards).sum()/len(episode.rewards)))
        if len(self.buffer) > self.max_len:
            delete_ind = np.random.randint(len(self.buffer))
            del self.buffer[delete_ind]
            self.weight = np.delete(self.weight, delete_ind)

    def sample(self):
        #return np.random.choice(self.buffer, p=self.weight/self.weight.sum())
        return np.random.choice(self.buffer, p=.1/len(self.buffer)+.9*(self.weight/self.weight.sum()))

    #@property
    def trainable(self):
        if len(self.buffer) > 32:
            return True
        else:
            return False