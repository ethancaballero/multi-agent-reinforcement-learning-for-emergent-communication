import os
import cv2
from gym.spaces.box import Box
import numpy as np
import gym
from gym import spaces
import logging
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode
import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
#universe.configure_logging()
universe.configure_logging(False)

from skimage.color import rgb2gray

os.environ['OPENAI_REMOTE_VERBOSE'] = '0'

def create_env(env_id, client_id, remotes, **kwargs):
    if env_id.find("Catcher-v0") != -1:
        import gym_ple
        import os
        os.putenv('SDL_VIDEODRIVER', 'fbcon')
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        return create_ple_env(env_id, **kwargs)

    spec = gym.spec(env_id)

    if spec.tags.get('flashgames', False):
        return create_flash_env(env_id, client_id, remotes, **kwargs)
    elif spec.tags.get('atari', False) and spec.tags.get('vnc', False):
        return create_vncatari_env(env_id, client_id, remotes, **kwargs)
    else:
        # Assume atari.
        assert "." not in env_id  # universe environments have dots in names.
        return create_atari_env(env_id, **kwargs)

def create_flash_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)

    reg = universe.runtime_spec('flashgames').server_registry
    height = reg[env_id]["height"]
    width = reg[env_id]["width"]
    env = CropScreen(env, height, width, 84, 18)
    env = FlashRescale(env)

    #env = NormalizedEnv(env)

    keys = ['left', 'right', 'up', 'down', 'x']
    if env_id == 'flashgames.NeonRace-v0':
        # Better key space for this game.
        keys = ['left', 'right', 'up', 'left up', 'right up', 'down', 'up x']
    if env_id == 'flashgames.LuxUltimate-v0':
        keys = ['a', 's', 'd']
    if env_id == 'flashgames.RetroRunner-v0':
        keys = ['up', 'right']
    if env_id == 'flashgames.RunRamRun-v0-v0':
        keys = ['up', 'down']
    logger.info('create_flash_env(%s): keys=%s', env_id, keys)

    env = DiscreteToFixedKeysVNCActions(env, keys)
    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    if "multi_obs_settings" in _.keys():
        _multi = [env]+list(_["multi_obs_settings"])
        #env = SplitScreen(*_multi)
        env = SplitScreen(*_multi)

    #env = Unvectorize(env)
    if not "no_config" in _.keys():
        env.configure(fps=5.0, remotes=remotes, start_timeout=15 * 60, client_id=client_id,
                      vnc_driver='go', vnc_kwargs={
                        'encoding': 'tight', 'compress_level': 0,
                        'fine_quality_level': 50, 'subsample_level': 3},
                        replace_on_crash=True)  
    return env

def create_vncatari_env(env_id, client_id, remotes, **_):
    env = gym.make(env_id)
    env = Vision(env)
    env = Logger(env)
    env = BlockingReset(env)
    env = GymCoreAction(env)
    env = AtariRescale42x42(env)

    #env = NormalizedEnv(env)

    env = EpisodeID(env)
    env = DiagnosticsInfo(env)
    if "multi_obs_settings" in _.keys():
        _multi = [env]+list(_["multi_obs_settings"])
        env = [env, SplitScreen(*_multi)]

    #env = Unvectorize(env)
    logger.info('Connecting to remotes: %s', remotes)
    fps = env.metadata['video.frames_per_second']
    if not "no_config" in _.keys():
        env.configure(remotes=remotes, start_timeout=15 * 60, fps=fps, client_id=client_id)
    return env

def create_atari_env(env_id, **_):
    env = gym.make(env_id)
    env = Vectorize(env)
    #env = AtariRescale42x42(env)
    env = AtariRescale80x80(env)
    env = NormalizedEnv(env)
    env = DiagnosticsInfo(env)
    if "multi_obs_settings" in _.keys():
        _multi = [env]+list(_["multi_obs_settings"])
        env = SplitScreen(*_multi)
        #env = [env, SplitScreen(*_multi)]

    #env = Unvectorize(env)
    return env

def create_ple_env(env_id, **_):
    env = gym.make(env_id)
    env = Vectorize(env)
    env = PleRescale(env)
    #env = DiagnosticsInfo(env)
    #env = NormalizedEnv(env)
    #env = Unvectorize(env)
    if "multi_obs_settings" in _.keys():
        _multi = [env]+list(_["multi_obs_settings"])
        env = SplitScreen(*_multi)
    #env = gym.wrappers.Monitor(env, '/tmp/catcher-experiment-1')
    return env

def DiagnosticsInfo(env, *args, **kwargs):
    return vectorized.VectorizeFilter(env, DiagnosticsInfoI, *args, **kwargs)

class DiagnosticsInfoI(vectorized.Filter):
    def __init__(self, log_interval=503):
    #def __init__(self, log_interval=1):
        super(DiagnosticsInfoI, self).__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._local_t = 0
        self._log_interval = log_interval
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        self._num_vnc_updates = 0
        self._last_episode_id = -1

    def _after_reset(self, observation):
        logger.info('Resetting environment')
        self._episode_reward = 0
        self._episode_length = 0
        self._all_rewards = []
        return observation

    def _after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._local_t += 1
        if info.get("stats.vnc.updates.n") is not None:
            self._num_vnc_updates += info.get("stats.vnc.updates.n")

        if self._local_t % self._log_interval == 0:
            cur_time = time.time()
            elapsed = cur_time - self._last_time
            fps = self._log_interval / elapsed
            self._last_time = cur_time
            cur_episode_id = info.get('vectorized.episode_id', 0)
            to_log["diagnostics/fps"] = fps
            if self._last_episode_id == cur_episode_id:
                to_log["diagnostics/fps_within_episode"] = fps
            self._last_episode_id = cur_episode_id
            if info.get("stats.gauges.diagnostics.lag.action") is not None:
                to_log["diagnostics/action_lag_lb"] = info["stats.gauges.diagnostics.lag.action"][0]
                to_log["diagnostics/action_lag_ub"] = info["stats.gauges.diagnostics.lag.action"][1]
            if info.get("reward.count") is not None:
                to_log["diagnostics/reward_count"] = info["reward.count"]
            if info.get("stats.gauges.diagnostics.clock_skew") is not None:
                to_log["diagnostics/clock_skew_lb"] = info["stats.gauges.diagnostics.clock_skew"][0]
                to_log["diagnostics/clock_skew_ub"] = info["stats.gauges.diagnostics.clock_skew"][1]
            if info.get("stats.gauges.diagnostics.lag.observation") is not None:
                to_log["diagnostics/observation_lag_lb"] = info["stats.gauges.diagnostics.lag.observation"][0]
                to_log["diagnostics/observation_lag_ub"] = info["stats.gauges.diagnostics.lag.observation"][1]

            if info.get("stats.vnc.updates.n") is not None:
                to_log["diagnostics/vnc_updates_n"] = info["stats.vnc.updates.n"]
                to_log["diagnostics/vnc_updates_n_ps"] = self._num_vnc_updates / elapsed
                self._num_vnc_updates = 0
            if info.get("stats.vnc.updates.bytes") is not None:
                to_log["diagnostics/vnc_updates_bytes"] = info["stats.vnc.updates.bytes"]
            if info.get("stats.vnc.updates.pixels") is not None:
                to_log["diagnostics/vnc_updates_pixels"] = info["stats.vnc.updates.pixels"]
            if info.get("stats.vnc.updates.rectangles") is not None:
                to_log["diagnostics/vnc_updates_rectangles"] = info["stats.vnc.updates.rectangles"]
            if info.get("env_status.state_id") is not None:
                to_log["diagnostics/env_state_id"] = info["env_status.state_id"]

        if reward is not None:
            self._episode_reward += reward
            if observation is not None:
                self._episode_length += 1
            self._all_rewards.append(reward)

        if done:
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            self._episode_reward = 0
            self._episode_length = 0
            self._all_rewards = []

        return observation, reward, done, to_log

def _process_frame42(frame):
    frame = frame[34:34+160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    #frame = np.reshape(frame, [42, 42, 1])
    frame = np.reshape(frame, [1, 42, 42])
    return frame

def _process_frame80(frame):
    frame = frame[34:34 + 160, :160]
    frame = cv2.resize(rgb2gray(frame), (80, 80))
    frame = cv2.resize(frame, (80, 80))
    frame = np.reshape(frame, [1, 80, 80])
    return frame

class AtariRescale42x42(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42, self).__init__(env)
        #self.observation_space = Box(0.0, 1.0, [42, 42, 1])
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def _observation(self, observation_n):
        return [_process_frame42(observation) for observation in observation_n]

class AtariRescale80x80(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale80x80, self).__init__(env)
        #self.observation_space = Box(0.0, 1.0, [80, 80, 1])
        self.observation_space = Box(0.0, 1.0, [1, 80, 80])

    def _observation(self, observation_n):
        return [_process_frame80(observation) for observation in observation_n]

def _process_frame_flash(frame):
    frame = cv2.resize(frame, (200, 128))
    frame = frame.mean(2).astype(np.float32)
    frame *= (1.0 / 255.0)
    #frame = np.reshape(frame, [128, 200, 1])
    frame = np.reshape(frame, [1, 128, 200])
    return frame

class FlashRescale(vectorized.ObservationWrapper):
    def __init__(self, env=None):
        super(FlashRescale, self).__init__(env)
        #self.observation_space = Box(0.0, 1.0, [128, 200, 1])
        self.observation_space = Box(0.0, 1.0, [1, 128, 200])

    def _observation(self, observation_n):
        return [_process_frame_flash(observation) for observation in observation_n]

def _process_frame_ple(frame):
    #converts to grayscale
    frame = 0.21 * frame[:, :, 0] + 0.72 * \
        frame[:, :, 1] + 0.07 * frame[:, :, 2]

    frame = frame.astype(np.float32)
    frame = np.reshape(frame, [1, 64, 64])
    return frame

class PleRescale(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PleRescale, self).__init__(env)
        #self.observation_space = Box(0.0, 1.0, [64, 64, 1])
        self.observation_space = Box(0.0, 1.0, [1, 64, 64])

    def _observation(self, observation_n):
        return [_process_frame_ple(observation) for observation in observation_n]

class FixedKeyState(object):
    def __init__(self, keys):
        self._keys = [keycode(key) for key in keys]
        self._down_keysyms = set()

    def apply_vnc_actions(self, vnc_actions):
        for event in vnc_actions:
            if isinstance(event, vnc_spaces.KeyEvent):
                if event.down:
                    self._down_keysyms.add(event.key)
                else:
                    self._down_keysyms.discard(event.key)

    def to_index(self):
        action_n = 0
        for key in self._down_keysyms:
            if key in self._keys:
                # If multiple keys are pressed, just use the first one
                action_n = self._keys.index(key) + 1
                break
        return action_n

class DiscreteToFixedKeysVNCActions(vectorized.ActionWrapper):
    """
    Define a fixed action space. Action 0 is all keys up. Each element of keys can be a single key or a space-separated list of keys
    For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right'])
    will have 3 actions: [none, left, right]
    You can define a state with more than one key down by separating with spaces. For example,
       e=DiscreteToFixedKeysVNCActions(e, ['left', 'right', 'space', 'left space', 'right space'])
    will have 6 actions: [none, left, right, space, left space, right space]
    """
    def __init__(self, env, keys):
        super(DiscreteToFixedKeysVNCActions, self).__init__(env)

        self._keys = keys
        self._generate_actions()
        self.action_space = spaces.Discrete(len(self._actions))

    def _generate_actions(self):
        self._actions = []
        uniq_keys = set()
        for key in self._keys:
            for cur_key in key.split(' '):
                uniq_keys.add(cur_key)

        for key in [''] + self._keys:
            split_keys = key.split(' ')
            cur_action = []
            for cur_key in uniq_keys:
                cur_action.append(vnc_spaces.KeyEvent.by_name(cur_key, down=(cur_key in split_keys)))
            self._actions.append(cur_action)
        self.key_state = FixedKeyState(uniq_keys)

    def _action(self, action_n):
        # Each action might be a length-1 np.array. Cast to int to
        # avoid warnings.
        return [self._actions[int(action)] for action in action_n]

class CropScreen(vectorized.ObservationWrapper):
    """Crops out a [height]x[width] area starting from (top,left) """
    def __init__(self, env, height, width, top=0, left=0):
        super(CropScreen, self).__init__(env)
        self.height = height
        self.width = width
        self.top = top
        self.left = left
        self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        return [ob[self.top:self.top+self.height, self.left:self.left+self.width, :] if ob is not None else None
                for ob in observation_n]

class SplitScreen(vectorized.ObservationWrapper):
    """
    Splits the observation of the environment.
    split_ratio: where split occurs
    split_pad: how much to pad splits to prevent observation leakage between agents' observations
    """
    def __init__(self, env, split_ratio, split_pad=0.0, split_hor_line=True, channels=1, channel_first=True):
        super(SplitScreen, self).__init__(env)
        assert (split_ratio + split_pad <= 1.0), "split_ratio + split_pad cannot be > 1.0"
        self.channel_first = channel_first
        self.channels = channels
        self.split_hor_line = split_hor_line
        self.top = 0
        self.left = 0
        if self.channel_first==True:
            self.height = self.observation_space.shape[1]
            self.width = self.observation_space.shape[2]
            if split_hor_line==True:
                self.split_point1 = int(split_ratio * self.height)
                self.split_point2 = int((split_ratio + split_pad) * self.height)
                self.observation_space_omni = self.observation_space
                self.observation_space = [Box(0, 255, shape=(self.channels, self.split_point1, self.width)), Box(0, 255, shape=(self.channels, self.height-self.split_point2, self.width))]
            else:
                self.split_point1 = int(split_ratio * self.width)
                self.split_point2 = int((split_ratio + split_pad) * self.width)
                self.observation_space_omni = self.observation_space
                self.observation_space = [Box(0, 255, shape=(self.channels, self.height, self.split_point1)), Box(0, 255, shape=(self.channels, self.height, self.width-self.split_point2))]
        else:
            self.height = self.observation_space.shape[0]
            self.width = self.observation_space.shape[1]
            if split_hor_line==True:
                self.split_point1 = int(split_ratio * self.height)
                self.split_point2 = int((split_ratio + split_pad) * self.height)
                self.observation_space_omni = self.observation_space
                self.observation_space = [Box(0, 255, shape=(self.split_point1, self.width, self.channels)), Box(0, 255, shape=(self.height-self.split_point2, self.width, self.channels))]
            else:
                self.split_point1 = int(split_ratio * self.width)
                self.split_point2 = int((split_ratio + split_pad) * self.width)
                self.observation_space_omni = self.observation_space
                self.observation_space = [Box(0, 255, shape=(self.height, self.split_point1, self.channels)), Box(0, 255, shape=(self.height, self.width-self.split_point2, self.channels))]

    ''' # version without omnicritic
    def _observation(self, observation_n):
        if observation_n[0] is not None:
            if self.channel_first==True:
                if self.split_hor_line==True:
                    return [observation_n[0][:, self.top:self.top+self.split_point1, self.left:self.left+self.width], 
                            observation_n[0][:, self.top+self.split_point2:self.top+self.height, self.left:self.left+self.width]]
                else:
                    return [observation_n[0][:, self.top:self.top+self.height, self.left:self.left+self.split_point1], 
                            observation_n[0][:, self.top:self.top+self.height, self.left+self.split_point2:self.left+self.width]]
            else:
                if self.split_hor_line==True:
                    return [observation_n[0][self.top:self.top+self.split_point1, self.left:self.left+self.width, :], 
                            observation_n[0][self.top+self.split_point2:self.top+self.height, self.left:self.left+self.width, :]]
                else:
                    return [observation_n[0][self.top:self.top+self.height, self.left:self.left+self.split_point1, :], 
                            observation_n[0][self.top:self.top+self.height, self.left+self.split_point2:self.left+self.width, :]]
        else:
            return [None, None]
            #'''

    def _observation(self, observation_n):
        if observation_n[0] is not None:
            if self.channel_first==True:
                if self.split_hor_line==True:
                    return [observation_n[0], [observation_n[0][:, self.top:self.top+self.split_point1, self.left:self.left+self.width], 
                            observation_n[0][:, self.top+self.split_point2:self.top+self.height, self.left:self.left+self.width]]]
                else:
                    return [observation_n[0], [observation_n[0][:, self.top:self.top+self.height, self.left:self.left+self.split_point1], 
                            observation_n[0][:, self.top:self.top+self.height, self.left+self.split_point2:self.left+self.width]]]
            else:
                if self.split_hor_line==True:
                    return [observation_n[0], [observation_n[0][self.top:self.top+self.split_point1, self.left:self.left+self.width, :], 
                            observation_n[0][self.top+self.split_point2:self.top+self.height, self.left:self.left+self.width, :]]]
                else:
                    return [observation_n[0], [observation_n[0][self.top:self.top+self.height, self.left:self.left+self.split_point1, :], 
                            observation_n[0][self.top:self.top+self.height, self.left+self.split_point2:self.left+self.width, :]]]
        else:
            return [None, [None, None]]

class NormalizedEnv(vectorized.ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8) for observation in observation_n]
