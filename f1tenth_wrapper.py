from scipy.spatial import KDTree
import gym
import os
import numpy as np
import pandas as pd
from gym.spaces import Box


class F110SB3Wrapper(gym.Wrapper):
    def __init__(self, env, start_pose, track_path, use_raceline=False):
        super().__init__(env)
        self.env = env
        self.start_pose = start_pose
        self.track_path = track_path
        self.use_raceline = use_raceline

        # --- Raceline ---
        self.raceline_xy = None
        self.raceline_psi = None
        self.raceline_vx = None
        self.raceline_kdtree = None
        self.raceline_curvature = None
        self.progress = None
        if use_raceline:
            csv_files = [f for f in os.listdir(track_path) if f.endswith("_raceline.csv")]
            if csv_files:
                df = pd.read_csv(os.path.join(track_path, csv_files[0]), skiprows=2, sep=';')
                df.columns = df.columns.str.strip()
                self.raceline_xy = np.vstack([df["x_m"].values, df["y_m"].values]).T
                # self.raceline_psi = df["psi_rad"].values
                self.raceline_vx = df["vx_mps"].values
                self.raceline_kdtree = KDTree(self.raceline_xy)
                if "# s_m" in df.columns:
                    self.progress = df["# s_m"].values
                else:
                    diffs = np.diff(self.raceline_xy, axis=0)      # shape (N-1, 2)
                    dists = np.linalg.norm(diffs, axis=1)            # Euclidean distances between consecutive points
                    self.progress = np.zeros(len(self.raceline_xy))
                    self.progress[1:] = np.cumsum(dists)


        # --- Centerline ---
        self.centerline_xy = None
        self.centerline_curvature = None
        csv_files = [f for f in os.listdir(track_path) if f.endswith("_centerline.csv")]
        if csv_files:
            df = pd.read_csv(os.path.join(track_path, csv_files[0]), sep=',')
            df.columns = df.columns.str.strip()
            x = df["# x_m"].values
            y = df["y_m"].values
            self.centerline_xy = np.vstack([x, y]).T
            if self.progress is None:
                diffs = np.diff(self.centerline_xy, axis=0)      # shape (N-1, 2)
                dists = np.linalg.norm(diffs, axis=1)            # Euclidean distances between consecutive points
                self.progress = np.zeros(len(self.centerline_xy))
                self.progress[1:] = np.cumsum(dists)   


        # --- Action space ---
        self.action_space = Box(low=np.array([-np.pi, 0.0], dtype=np.float32),
                                high=np.array([np.pi, 5.0], dtype=np.float32),
                                dtype=np.float32)

        # --- Observation space ---
        obs_dict, _, _, _ = self.env.reset(poses=np.zeros((self.env.num_agents, 3)))
        self.obs_keys = list(obs_dict.keys())
        spaces = {}
        for k in self.obs_keys:
            val = np.array(obs_dict[k], dtype=np.float32).flatten()
            spaces[k] = Box(low=-np.inf*np.ones(val.shape, dtype=np.float32),
                            high=np.inf*np.ones(val.shape, dtype=np.float32),
                            dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

        # Track previous state
        self.last_pos = None
        self.prev_steer = None
    def reset(self, random_start=True, **kwargs):
        if 'poses' in kwargs:
            poses = kwargs['poses']
            # print(f"Resetting with provided poses: {poses}")
        elif self.centerline_xy is not None and random_start:
            # print("Resetting with random start on centerline")
            s_idx = np.random.randint(0, len(self.centerline_xy))
            x, y = self.centerline_xy[s_idx]
            psi = 0.0
            if s_idx < len(self.centerline_xy) - 1:
                dx = self.centerline_xy[s_idx + 1, 0] - x
                dy = self.centerline_xy[s_idx + 1, 1] - y
                psi = np.arctan2(dy, dx)
            poses = np.array([[x, y, psi]], dtype=np.float32)
        else:
            poses = kwargs.get("poses", self.start_pose)

        obs_dict, _, _, _ = self.env.reset(poses=poses)
        ego_idx = obs_dict['ego_idx']
        self.last_pos = np.array([obs_dict['poses_x'][ego_idx], obs_dict['poses_y'][ego_idx]])
        self.prev_steer = None

        obs_dict = {k: np.array(v, dtype=np.float32).flatten() for k, v in obs_dict.items()}
        return obs_dict
    
    def step(self, action):
        action = np.atleast_2d(action)
        steer = float(action[0, 0])
        speed = float(action[0, 1])

        obs, _, done, info = self.env.step(action)
        ego_idx = obs['ego_idx']
        ego_pos = np.array([obs['poses_x'][ego_idx], obs['poses_y'][ego_idx]])
        ego_theta = obs['poses_theta'][ego_idx]

        # --- Find closest point on raceline or centerline ---
        if self.use_raceline and self.raceline_kdtree is not None:
            dist, idx = self.raceline_kdtree.query(ego_pos)
            # calculate reference heading
            dx = self.raceline_xy[min(idx+1, len(self.raceline_xy)-1),0] - self.raceline_xy[idx,0]
            dy = self.raceline_xy[min(idx+1, len(self.raceline_xy)-1),1] - self.raceline_xy[idx,1]
            psi_ref = np.arctan2(dy, dx)
            # check first psi_ref
            # print(psi_ref)
            vx_ref = self.raceline_vx[idx]
        else:
            kdtree = KDTree(self.centerline_xy)
            dist, idx = kdtree.query(ego_pos)
            dx = self.centerline_xy[min(idx+1, len(self.centerline_xy)-1),0] - self.centerline_xy[idx,0]
            dy = self.centerline_xy[min(idx+1, len(self.centerline_xy)-1),1] - self.centerline_xy[idx,1]
            psi_ref = np.arctan2(dy, dx)
            vx_ref = 2.0

        # --- Forward progress reward along track tangent ---
        track_vec = np.array([np.cos(psi_ref), np.sin(psi_ref)])
        progress = np.dot(ego_pos - self.last_pos, track_vec) if self.last_pos is not None else 0.0
        self.last_pos = ego_pos
        progress_reward = np.clip(progress, 0, 0.3)

        # --- Collision penalty ---
        collision_penalty = -1.0 if obs["collisions"][ego_idx] else 0.0

        # --- Wall centering reward ---
        scan = obs['scans'][ego_idx].flatten()
        left_dist = np.mean(scan[:max(1, len(scan)//10)])
        right_dist = np.mean(scan[-max(1, len(scan)//10):])
        wall_centering_reward = 0.05 * np.exp(-5 * abs(left_dist - right_dist))

        min_dist = np.min(scan)
        wall_prox_penalty = -0.05 * max(0.0, 0.3 - min_dist)

        # --- Steering smoothness penalty ---
        # if self.prev_steer is None:
        #     self.prev_steer = steer
        # steer_penalty = -0.01 * abs(steer - self.prev_steer) * max(0.2, 1.0 - 5 * curvature)
        # self.prev_steer = steer

        # Progress reward using self.progress if available
        if self.use_raceline and self.raceline_kdtree is not None and self.progress is not None:
            prog = self.progress[idx]
            progress_reward_status = prog / 100.0  # Scale as needed
        # --- Raceline reward ---
        raceline_reward = 0.0
        if self.use_raceline and self.raceline_kdtree is not None:
            dist_reward = 0.2 * np.exp(-5 * dist**2)
            heading_diff = abs(np.arctan2(np.sin(ego_theta - psi_ref), np.cos(ego_theta - psi_ref)))
            heading_reward = 0.1 * np.exp(-5 * heading_diff**2)
            # velocity_diff = abs(speed - vx_ref)
            # velocity_reward = 0.05 * np.exp(-0.5 * velocity_diff**2)
            raceline_reward = dist_reward + heading_reward #+ velocity_reward
            
        speed_reward = 0.05*speed
        reward = (
            progress_reward +
            progress_reward_status +
            # wall_centering_reward +
            # wall_prox_penalty +
            # steer_penalty +
            speed_reward +
            raceline_reward +
            collision_penalty
        )

        obs_dict = {k: np.array(v, dtype=np.float32).flatten() for k, v in obs.items()}
        return obs_dict, reward, done, info