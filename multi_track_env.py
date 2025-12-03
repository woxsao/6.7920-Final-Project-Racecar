import gym
import numpy as np
import random
from utils import get_initial_pose

class MultiTrackEnv(gym.Env):
    def __init__(self, track_paths, make_env_fn):
        super().__init__()
        self.track_paths = track_paths
        self.make_env_fn = make_env_fn
        self.env = None
        self.action_space = None
        self.observation_space = None
        self._load_new_track()

    def _load_new_track(self):
        track = random.choice(self.track_paths)
        # print(f"Loading track: {track}")
        self.env = self.make_env_fn(track)
        self.track_initial_pose = get_initial_pose(track)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, **kwargs):
        self._load_new_track()
        return self.env.reset(poses=self.track_initial_pose, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info
