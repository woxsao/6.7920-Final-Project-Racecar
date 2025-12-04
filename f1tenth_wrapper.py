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

        # --- Raceline data (optional) ---
        self.raceline_xy = None
        self.raceline_vx = None
        self.raceline_kdtree = None

        # --- Centerline data ---
        self.centerline_xy = None
        self.centerline_kdtree = None

        # --- Progress and lap tracking ---
        self.progress = None        # s_m along raceline/centerline
        self.last_prog = 0.0        # previous s_m for incremental reward
        self.lap_count = 0

        # --- Load raceline if requested ---
        if use_raceline:
            csv_files = [f for f in os.listdir(track_path) if f.endswith("_raceline.csv")]
            if csv_files:
                df = pd.read_csv(os.path.join(track_path, csv_files[0]), skiprows=2, sep=';')
                df.columns = df.columns.str.strip()
                self.raceline_xy = np.vstack([df["x_m"].values, df["y_m"].values]).T
                self.raceline_vx = df["vx_mps"].values if "vx_mps" in df.columns else None
                # print(f'Using raceline! xy : {self.raceline_xy} | vx : {self.raceline_vx}')
                
                self.raceline_kdtree = KDTree(self.raceline_xy)
                # progress: use # s_m if available otherwise compute arc-length
                if "# s_m" in df.columns:
                    self.progress = df["# s_m"].values.copy()
                else:
                    diffs = np.diff(self.raceline_xy, axis=0)
                    dists = np.linalg.norm(diffs, axis=1)
                    p = np.zeros(len(self.raceline_xy))
                    p[1:] = np.cumsum(dists)
                    self.progress = p

        # --- Load centerline (used as fallback) ---
        csv_files = [f for f in os.listdir(track_path) if f.endswith("_centerline.csv")]
        if csv_files:
            df = pd.read_csv(os.path.join(track_path, csv_files[0]), sep=',')
            df.columns = df.columns.str.strip()
            x = df["# x_m"].values
            y = df["y_m"].values
            self.centerline_xy = np.vstack([x, y]).T
            self.centerline_kdtree = KDTree(self.centerline_xy)
            # if progress not set by raceline, compute from centerline
            if self.progress is None:
                diffs = np.diff(self.centerline_xy, axis=0)
                dists = np.linalg.norm(diffs, axis=1)
                p = np.zeros(len(self.centerline_xy))
                p[1:] = np.cumsum(dists)
                self.progress = p

        # --- Action space ---
        self.action_space = Box(
            low=np.array([-np.pi, 0.0], dtype=np.float32),
            high=np.array([np.pi, 5.0], dtype=np.float32), # FIXME is this supposed to be capped?
            dtype=np.float32,
        )

        # --- Observation space (derived from env) ---
        obs_dict, _, _, _ = self.env.reset(poses=np.zeros((self.env.num_agents, 3)))
        self.obs_keys = list(obs_dict.keys())
        spaces = {}
        for k in self.obs_keys:
            val = np.array(obs_dict[k], dtype=np.float32).flatten()
            spaces[k] = Box(low=-np.inf * np.ones(val.shape, dtype=np.float32),
                            high=np.inf * np.ones(val.shape, dtype=np.float32),
                            dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)

        # --- State trackers ---
        self.last_pos = None
        self.prev_steer = None

    def reset(self, random_start=True, **kwargs):
        # Accept 'poses' kwarg (used by your existing code)
        if 'poses' in kwargs:
            poses = kwargs['poses']
        elif self.centerline_xy is not None and random_start:
            # random start on centerline
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

        # initialize last_prog using nearest track point (if available)
        if self.progress is not None:
            pos = self.last_pos
            if self.use_raceline and self.raceline_kdtree is not None:
                _, idx = self.raceline_kdtree.query(pos)
            else:
                _, idx = self.centerline_kdtree.query(pos)
            self.last_prog = float(self.progress[idx])
        else:
            self.last_prog = 0.0

        obs_dict = {k: np.array(v, dtype=np.float32).flatten() for k, v in obs_dict.items()}
        return obs_dict

    def step(self, action):
        action = np.atleast_2d(action)
        steer_action = float(action[0, 0])
        speed_action = float(action[0, 1])

        obs, _, done, info = self.env.step(action)
        
        ego_idx = obs['ego_idx']
        ego_pos = np.array([obs['poses_x'][ego_idx], obs['poses_y'][ego_idx]])
        ego_theta = obs['poses_theta'][ego_idx]

        # # --- Find closest reference point (raceline if available else centerline) ---
        if self.use_raceline and self.raceline_kdtree is not None:
            dist, idx = self.raceline_kdtree.query(ego_pos)
            ref_xy = self.raceline_xy
        else:
            dist, idx = self.centerline_kdtree.query(ego_pos)
            ref_xy = self.centerline_xy

        # compute reference heading (tangent) at idx
        next_idx = min(idx + 1, len(ref_xy) - 1)
        dx = ref_xy[next_idx, 0] - ref_xy[idx, 0]
        dy = ref_xy[next_idx, 1] - ref_xy[idx, 1]
        psi_ref = np.arctan2(dy, dx)

        # # --- Forward incremental progress reward (along track tangent) ---
        # track_vec = np.array([np.cos(psi_ref), np.sin(psi_ref)])
        # step_delta = ego_pos - self.last_pos if self.last_pos is not None else np.array([0.0, 0.0])
        # forward_progress = float(np.dot(step_delta, track_vec))
        # self.last_pos = ego_pos.copy()
        # progress_reward = float(np.clip(forward_progress, 0.0, 1.0))  # reward for moving forward

        # # --- Incremental s_m reward (stable lap progress) ---
        # progress_reward_inc = 0.0
        # if self.progress is not None:
        #     prog = float(self.progress[idx])
        #     prog_delta = prog - self.last_prog
        #     # handle wrap-around if negative (optional)
        #     if prog_delta < -0.5 * self.progress[-1]:
        #         # agent likely wrapped to start (lap)
        #         prog_delta = (prog + self.progress[-1]) - self.last_prog
        #     prog_delta = max(prog_delta, 0.0)
        #     progress_reward_inc = float(np.clip(prog_delta, 0.0, 2.0))  # cap to avoid explosion
        #     self.last_prog = prog

        # # --- Collision penalty ---
        # collision_penalty = -1.0 if obs["collisions"][ego_idx] else 0.0

        # # --- Scan-based signals ---
        scan = obs['scans'][ego_idx].flatten()
        # n = len(scan)
        # # left / right coarse averages
        # left_dist = np.mean(scan[: max(1, n // 6)])
        # right_dist = np.mean(scan[- max(1, n // 6):])
        # front_dist = np.mean(scan[n//3: -n//3]) if n > 6 else np.mean(scan)

        # # wall proximity penalty (soft)
        min_dist = float(np.min(scan))
        # wall_prox_penalty = -0.02 * max(0.0, 0.15 - min_dist)

        # # scan-based turn cue (lookahead sectors)
        # look_ahead = max(1, n // 4)
        # left_ahead = np.mean(scan[:look_ahead])
        # right_ahead = np.mean(scan[-look_ahead:])
        # turn_signal = np.tanh((right_ahead - left_ahead) / 2.0)  # in [-1,1]

        # # only reward steering when turn_signal is significant
        # scan_turn_reward = 0.0
        # if abs(turn_signal) > 0.06:
        #     # encourage steering toward more open side (scaled)
        #     scan_turn_reward = 0.06 * turn_signal * steer_action

        # # --- Raceline / heading reward (optional) ---
        # raceline_reward = 0.0
        # if self.use_raceline and self.raceline_kdtree is not None:
        #     heading_diff = abs(np.arctan2(np.sin(ego_theta - psi_ref), np.cos(ego_theta - psi_ref)))
        #     raceline_reward = 0.15 * np.exp(-8.0 * heading_diff**2)  # encourage alignment

        # # --- Speed shaping (encourage moderate speed, penalize extreme in corners) ---
        # # simple reward proportional to speed (keeps agent moving)
        # # speed_reward = 0.03 * speed_action
        # actual_speed = np.sqrt(obs['linear_vels_x'][0]**2 + obs['linear_vels_y'][0]**2)
        # speed_reward = 0.01 * actual_speed
        
        # if actual_speed < 0.1:
        #     speed_reward -= 0.05

        # # Optional speed penalty if front is too close
        # speed_penalty = 0.0
        # if front_dist < 0.5:
        #     speed_penalty = -0.05 * max(0.0, 0.5 - front_dist)

        # # --- Lap completion bonus (small, on first detection) ---
        # lap_bonus = 0.0
        # if "lap_counts" in obs and obs["lap_counts"][ego_idx] > self.lap_count:
        #     self.lap_count = int(obs["lap_counts"][ego_idx])
        #     lap_bonus = 5.0

        # # --- Total reward (balanced) ---
        # reward = (
        #     progress_reward +
        #     progress_reward_inc +
        #     speed_reward +
        #     raceline_reward +
        #     scan_turn_reward +
        #     wall_prox_penalty +
        #     speed_penalty +
        #     collision_penalty +
        #     lap_bonus
        # )
        
        # 1. Progress (ONLY when moving)
        progress_reward = 0.0
        actual_speed = np.sqrt(obs['linear_vels_x'][0]**2 + obs['linear_vels_y'][0]**2)
        
        if (actual_speed > 0.3 # car actually have to be moving, not just commanding some vel
            and self.progress is not None # have track data loaded (raceline or centerline)
        ):
            prog = float(self.progress[idx]) # arc length from start of track to closest point on track right now
            prog_delta = prog - self.last_prog
            
            if prog_delta < -0.5 * self.progress[-1]:
                prog_delta = (prog + self.progress[-1]) - self.last_prog
                
            progress_reward = float(np.clip(max(prog_delta, 0.0), 0.0, 0.5))
            self.last_prog = prog

        # 2. Speed (capped, only when not too close to walls)
        speed_reward = 0.0
        if min_dist > 0.3:
            speed_reward = 0.03 * min(actual_speed, 2.5)
        else:
            speed_reward = -0.1  # Too close, slow down!

        # 3. Stay on track
        track_penalty = -0.15 * min(dist, 1.5)

        # 4. Penalties
        collision_penalty = -5.0 if obs["collisions"][ego_idx] else 0.0
        wall_penalty = -0.5 * max(0.0, 0.25 - min_dist)

        # 5. Lap bonus 
        lap_bonus = 0.0
        if "lap_counts" in obs and obs["lap_counts"][ego_idx] > self.lap_count:
            self.lap_count = int(obs["lap_counts"][ego_idx])
            lap_bonus = 3.0

        # 6. Heading bonus
        heading_error = abs(np.arctan2(np.sin(ego_theta - psi_ref), 
                                    np.cos(ego_theta - psi_ref)))
        heading_reward = 0.1 * np.exp(-3.0 * heading_error**2)

        # Total
        reward = (
            progress_reward +
            speed_reward +
            # track_penalty +
            collision_penalty +
            # wall_penalty +
            lap_bonus + 
            heading_reward
        )

        # Convert observations to np.float32 flattened dict (same as before)
        obs_dict = {k: np.array(v, dtype=np.float32).flatten() for k, v in obs.items()}
        return obs_dict, float(reward), done, info
