from typing import Callable, OrderedDict
import gym
import numpy as np
import random
from f1tenth_wrapper import F110SB3Wrapper
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



class MultiTrackCurriculumEnv(gym.Env):
    '''
    Trains on multiple tracks following the given curriculum ({env : # train steps} dict)
    
    Once all tracks have been trained for the specified # of steps in order, the environment
    will randomly sample from the list of tracks in the curriculum
    '''
    def __init__(self, curriculum: OrderedDict, make_env_fn: Callable[[str], F110SB3Wrapper]):
        super().__init__()
        self.curriculum = curriculum
        self.curriculum_steps_completed = OrderedDict({track: 0 for track in self.curriculum.keys()})
        self.make_env_fn = make_env_fn
        
        self.current_track = None
        self.env = None
        self.action_space = None
        self.observation_space = None
        
        print(f"Initializing curriculum with tracks: {list(curriculum.keys())}")
        self._load_new_track()
    
    def _load_new_track(self):
        """Load a new track - called when switching to next curriculum stage"""
        # Select the first track that hasn't completed its budget
        track = next(
            (k for k in self.curriculum 
             if self.curriculum_steps_completed[k] < self.curriculum[k]), 
            None
        )
        
        if track is None:
            # All tracks complete, sample randomly
            track = random.choice(list(self.curriculum.keys()))
            print(f"Curriculum complete! Randomly sampled track: {track}")
        else:
            steps_done = self.curriculum_steps_completed.get(track, 0)
            steps_total = self.curriculum[track]
            print(f"Loading track: {track} ({steps_done}/{steps_total} steps)\n\t\tAll counters: {dict(self.curriculum_steps_completed)}")
        
        # Set current track FIRST
        self.current_track = track
        
        try:
            self.env = self.make_env_fn(track)
            self.track_initial_pose = get_initial_pose(track)
            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            print(f"\tSuccessfully loaded: {track}")
        except Exception as e:
            print(f"\tERROR loading track {track}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def reset(self, **kwargs):
        """Reset the environment, switching tracks if needed"""
        # Check if current track has completed its step budget
        if self.current_track is not None:
            steps_done = self.curriculum_steps_completed[self.current_track]
            steps_needed = self.curriculum[self.current_track]
            
            if steps_done >= steps_needed:
                print(f"\t✅ Track '{self.current_track}' completed ({steps_done}/{steps_needed} steps)!")
                self._load_new_track()  # Switch to next track
        elif self.current_track is None:
            # Safety: shouldn't happen, but reload if needed
            print("\t⚠️ WARNING: current_track is None in reset()!")
            self._load_new_track()
        
        return self.env.reset(poses=self.track_initial_pose, **kwargs)
    
    def step(self, action):
        """Execute one step, incrementing curriculum counter"""
        if self.current_track is None:
            raise RuntimeError("current_track is None! Wrapper not properly initialized.")
        
        obs, reward, done, info = self.env.step(action)
        
        # Increment step counter for current track
        self.curriculum_steps_completed[self.current_track] += 1
        
        # Log progress periodically
        total_steps = sum(self.curriculum_steps_completed.values())
        if total_steps % 1000 == 0:
            print(f"\tTotal steps: {total_steps:,} | Counters: {dict(self.curriculum_steps_completed)}")
        
        # Add curriculum metadata to info dict for logging
        info['curriculum/track'] = self.current_track
        info['curriculum/steps'] = self.curriculum_steps_completed[self.current_track]
        info['curriculum/progress'] = (
            self.curriculum_steps_completed[self.current_track] / 
            self.curriculum[self.current_track]
        )
        
        return obs, reward, done, info