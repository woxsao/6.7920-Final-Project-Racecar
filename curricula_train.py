import os
import yaml
import numpy as np
import pandas as pd
import gym
from argparse import Namespace
from stable_baselines3 import PPO
from f110_gym.envs.base_classes import Integrator
from f1tenth_wrapper import F110SB3Wrapper
from multi_track_env import MultiTrackEnv
import warnings
from utils import load_track_config, get_initial_pose, make_env_from_track, get_tracks_list
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
# -----------------------------
# Config
# -----------------------------
TRACKS_DIR = "./curricula"
TIMESTEP = 0.01
TIMESTEPS_PER_TRACK = 1_000_000
TRAIN_SPLIT = 1 # usually 0.8


# -----------------------------
# Main training function
# -----------------------------
def main():
    # Detect tracks
    track_paths = get_tracks_list(TRACKS_DIR)
    if not track_paths:
        raise RuntimeError("No tracks found in TRACKS_DIR")

    print(f"Found {len(track_paths)} tracks.")

    # Split into training/validation sets
    num_train = int(len(track_paths) * TRAIN_SPLIT)
    num_train = 1
    train_tracks = track_paths[:num_train]
    val_tracks = track_paths[num_train:]
    
    train_tracks = ['./curricula/0_circle']

    print(f"Training on {train_tracks} tracks, saving {len(val_tracks)} tracks for unseen validation.")

    # Save track splits
    with open("train_tracks.txt", "w") as f:
        for t in train_tracks: f.write(f"{t}\n")
    with open("val_tracks.txt", "w") as f:
        for t in val_tracks: f.write(f"{t}\n")

    # -----------------------------
    # Multi-track training environment
    # -----------------------------
    curriculum = yaml.safe_load(open("curriculum.yaml", "r"))
    x = lambda : MultiTrackEnv(train_tracks, lambda p: make_env_from_track(p, use_raceline=True))
    
    # train_env = make_vec_env(x, n_envs=8)
    train_env = x()
    train_env.reset()

    # -----------------------------
    # Stable-Baselines3 PPO agent
    # -----------------------------
    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tensorboard/f110_multi_track_ppo/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=512,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )

    # Train
    total_timesteps = TIMESTEPS_PER_TRACK * len(train_tracks)
    model.learn(total_timesteps, callback=ActionLoggingCallback())
    model.save("curricula_results/ppo_f110_0_circle")
    print("Training complete and saved!")


class ActionLoggingCallback(BaseCallback):
    """
    Logs agent actions to TensorBoard during training.

    Arguments:
    - log_freq: log action every N environment steps.
    """

    def __init__(self, log_freq=1, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.global_step = 0

    def _on_step(self) -> bool:
        """
        Called at each environment step when using on_policy algorithms,
        or at each rollout collection step for off_policy algorithms.
        """

        # Increment step counter
        self.global_step += 1

        # Access actions from the rollout buffer (always stores the *previous* step’s action)
        actions = self.locals.get("actions", None)

        # Some algorithms store actions inside env attributes instead
        if actions is None:
            # Vectorized env trick: gets last actions taken by env step()
            actions = getattr(self.training_env, "last_actions", None)

        if actions is None:
            return True  # can't log, skip

        # Sampled logging
        if self.global_step % self.log_freq == 0:
            actions = np.array(actions)

            # Handle both single env and VecEnv
            if actions.ndim == 1:
                # Example: [steer, throttle]
                for i, a in enumerate(actions):
                    self.logger.record(f"actions/action_{i}", a)
            else:
                # Example: multiple parallel envs
                for env_i in range(actions.shape[0]):
                    for a_i in range(actions.shape[1]):
                        self.logger.record(f"actions/env{env_i}_action_{a_i}", actions[env_i, a_i])

            # SB3 requires calling logger.dump at training frequency, but callbacks let SB3 handle it
        return True


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="f110_gym.envs.base_classes")
    main()
