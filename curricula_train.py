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

# -----------------------------
# Config
# -----------------------------
TRACKS_DIR = "./curricula"
TIMESTEP = 0.01
TIMESTEPS_PER_TRACK = 1_000_000
TRAIN_SPLIT = 0.8


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
    # num_train = int(len(track_paths) * TRAIN_SPLIT)
    num_train = 1
    train_tracks = track_paths[:num_train]
    val_tracks = track_paths[num_train:]

    print(f"Training on {len(train_tracks)} tracks, validating on {len(val_tracks)} tracks.")

    # Save track splits
    with open("train_tracks.txt", "w") as f:
        for t in train_tracks: f.write(f"{t}\n")
    with open("val_tracks.txt", "w") as f:
        for t in val_tracks: f.write(f"{t}\n")

    # -----------------------------
    # Multi-track training environment
    # -----------------------------
    train_env = MultiTrackEnv(train_tracks, lambda p: make_env_from_track(p, use_raceline=True))
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
    model.learn(total_timesteps=TIMESTEPS_PER_TRACK * len(train_tracks))
    model.save("curricula_results/ppo_f110_multi_track_use_scan")
    print("Training complete and saved!")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="f110_gym.envs.base_classes")
    main()
