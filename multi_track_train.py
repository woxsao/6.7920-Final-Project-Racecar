import os
import yaml
import numpy as np
import pandas as pd
import gym
from argparse import ArgumentParser
from stable_baselines3 import PPO
from f110_gym.envs.base_classes import Integrator
from f1tenth_wrapper import F110SB3Wrapper
from multi_track_env import MultiTrackEnv
import warnings
from utils import load_track_config, get_initial_pose, make_env_from_track, get_tracks_list
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

# -----------------------------
# Config
# -----------------------------
TRACKS_DIR = "./envs/f1tenth_racetracks"
TIMESTEP = 0.01
TIMESTEPS_PER_TRACK = 4_000_000
TRAIN_SPLIT = 0.8
TENSORBOARD_LOG = "./tensorboard/f110_multi_track_ppo/"

# -----------------------------
# Main training function
# -----------------------------
def main(model_path=None):
    # Detect tracks
    track_paths = get_tracks_list(TRACKS_DIR)
    if not track_paths:
        raise RuntimeError("No tracks found in TRACKS_DIR")

    print(f"Found {len(track_paths)} tracks.")

    # Split into training/validation sets
    num_train = max(1, int(len(track_paths) * TRAIN_SPLIT))
    # num_train = 5
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
    initial_pose = get_initial_pose(train_tracks[0])
    # train_env = MultiTrackEnv(train_tracks, lambda p: make_env_from_track(p, use_raceline=True))
    x = lambda : MultiTrackEnv(train_tracks, lambda p: make_env_from_track(p, use_raceline=True))
    train_env = make_vec_env(
        x,
        n_envs=4,
        vec_env_cls=DummyVecEnv,
    )
    train_env.reset()
    # -----------------------------
    # Stable-Baselines3 PPO agent
    # -----------------------------
    if model_path and os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model = PPO.load(model_path, env=train_env)
    else:
        print("Training from scratch")
        model = PPO(
            "MultiInputPolicy",
            train_env,
            verbose=1,
            tensorboard_log=TENSORBOARD_LOG,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
        )

    # -----------------------------
    # Train / Fine-tune
    # -----------------------------
    model.learn(total_timesteps=TIMESTEPS_PER_TRACK * len(train_tracks))

    # -----------------------------
    # Save
    # -----------------------------
    name = "ppo_f110_all_train"
    save_path = f"results/{name}_finetuned" if model_path else f"results/{name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Training complete. Model saved to {save_path}")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="f110_gym.envs.base_classes")

    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to pre-trained model (optional)")
    args = parser.parse_args()

    main(model_path=args.model)
