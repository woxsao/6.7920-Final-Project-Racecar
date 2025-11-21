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
        self.last_pos = None
        self.prev_distance_to_finish = None
    def reset(self, **kwargs):
        poses = kwargs.get("poses", self.start_pose)
        obs_dict, _, _, _ = self.env.reset(poses=poses)

        ego_idx = obs_dict['ego_idx']
        ego_pos = np.array([obs_dict['poses_x'][ego_idx], obs_dict['poses_y'][ego_idx]])
        finish_line = np.array([self.env.start_xs[ego_idx], self.env.start_ys[ego_idx]]) + 10 * self.env.start_rot[:, 0]

        # store distances for reward shaping
        self.last_pos = ego_pos
        self.prev_distance_to_finish = np.linalg.norm(finish_line - ego_pos)

        # build observation including goal vector
        goal_vec = finish_line - ego_pos
        obs_dict = {k: np.array(v, dtype=np.float32).flatten() for k, v in obs_dict.items()}
        obs_dict['goal_vec'] = goal_vec.astype(np.float32)

        return obs_dict

    def step(self, action):
        action = np.atleast_2d(action)  # ensure 2D for the simulator
        obs, _, done, info = self.env.step(action)

        ego_idx = obs['ego_idx']
        ego_pos = np.array([obs['poses_x'][ego_idx], obs['poses_y'][ego_idx]])

        if self.last_pos is None:
            self.last_pos = ego_pos

        # simple progress reward: encourage movement
        delta_pos = ego_pos - self.last_pos
        reward = np.linalg.norm(delta_pos)  # reward is distance moved since last step

        # collision penalty
        reward += -5.0 if obs['collisions'][ego_idx] else 0.0

        # update last position
        self.last_pos = ego_pos

        # add goal vector to observation (optional)
        finish_line = np.array([self.env.start_xs[ego_idx], self.env.start_ys[ego_idx]]) + 10 * self.env.start_rot[:, 0]
        goal_vec = finish_line - ego_pos

        obs_dict = {k: np.array(v, dtype=np.float32).flatten() for k, v in obs.items()}
        obs_dict['goal_vec'] = goal_vec.astype(np.float32)

        return obs_dict, reward, done, info




