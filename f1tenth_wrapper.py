import gym
import numpy as np
from gym.spaces import Box

class F110SB3Wrapper(gym.Wrapper):
    def __init__(self, env, start_pose):
        super().__init__(env)

        self.start_pose = start_pose
        self.env = env
        self.action_space = Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 5.0], dtype=np.float32),
            dtype=np.float32
        )
        obs_dict, _reward, _done, _info = self.env.reset(poses = np.zeros((self.env.num_agents, 3)))
        self.obs_keys = list(obs_dict.keys())
        """
        Observation key: ego_idx, shape: ()
        Observation key: scans, shape: (1, 1080)
        Observation key: poses_x, shape: (1,)
        Observation key: poses_y, shape: (1,)
        Observation key: poses_theta, shape: (1,)
        Observation key: linear_vels_x, shape: (1,)
        Observation key: linear_vels_y, shape: (1,)
        Observation key: ang_vels_z, shape: (1,)
        Observation key: collisions, shape: (1,)
        Observation key: lap_times, shape: (1,)
        Observation key: lap_counts, shape: (1,)
        """

        # self.observation_space is going to be a dict of {obs.keys: Box(low, high, shape)}
        spaces = {}
        for k in self.obs_keys:
            val = obs_dict[k]
            val = np.array([val]) if np.isscalar(val) else np.array(val, dtype=np.float32)
            spaces[k] = Box(low=-np.inf*np.ones(val.shape, dtype=np.float32),
                            high=np.inf*np.ones(val.shape, dtype=np.float32),
                            dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
    def reset(self, **kwargs):
        poses = kwargs.get("poses", self.start_pose)
        obs_dict, _, _, _ = self.env.reset(poses=poses)
        return {k: np.array(v, dtype=np.float32).flatten() for k, v in obs_dict.items()}

    def step(self, action):
        # Ensure correct shape (num_agents, 2)
        action = np.atleast_2d(action).astype(np.float32)
        obs_dict, reward, done, info = self.env.step(action)
        obs_out = {k: np.array(v, dtype=np.float32).flatten() for k, v in obs_dict.items()}
        # print("Action taken:", action)
        return obs_out, reward, done, info

