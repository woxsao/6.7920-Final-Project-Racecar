import numpy as np
import pandas as pd
import yaml
import os
from argparse import Namespace
from f110_gym.envs.base_classes import Integrator
from f1tenth_wrapper import F110SB3Wrapper
import gym


# -----------------------------
# Utilities
# -----------------------------
def load_track_config(yaml_path):
    """Load YAML track config into Namespace."""
    with open(yaml_path, "r") as f:
        return Namespace(**yaml.safe_load(f))

def get_initial_pose(track_path):
    """Extract initial pose from raceline/centerline CSV."""
    csv_files = [f for f in os.listdir(track_path) if f.endswith("_centerline.csv")]
    if not csv_files:
        # fallback
        return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

    df = pd.read_csv(os.path.join(track_path, csv_files[0]), sep=",")
    df.columns = df.columns.str.strip()
    x, y = df["# x_m"][0], df["y_m"][0]
    x1, y1 = df["# x_m"][1], df["y_m"][1]
    psi = np.arctan2(y1 - y, x1 - x)
    return np.array([[x, y, psi]], dtype=np.float32)

TIMESTEP = 0.01
def make_env_from_track(track_path, use_raceline=False):
    """Construct a wrapped F110 Gym environment from a track folder."""
    yaml_files = [f for f in os.listdir(track_path) if f.endswith(".yaml")]
    if not yaml_files:
        raise ValueError(f"No YAML file in {track_path}")

    conf = load_track_config(os.path.join(track_path, yaml_files[0]))
    initial_pose = get_initial_pose(track_path)
    # print(f"Initial pose for track {track_path}: {initial_pose}")
    map_name = conf.image[:-4]

    env = gym.make(
        "f110_gym:f110-v0",
        map=os.path.join(track_path, map_name),
        map_ext=".png",
        num_agents=1,
        timestep=TIMESTEP,
        integrator=Integrator.RK4,
    )

    # Wrap environment
    env = F110SB3Wrapper(env, start_pose=initial_pose, track_path=track_path, use_raceline=use_raceline)
    return env

def get_tracks_list(track_dir):
    """List all track directories with a YAML file."""
    return [os.path.join(track_dir, d) for d in os.listdir(track_dir)
            if os.path.isdir(os.path.join(track_dir, d)) and any(f.endswith(".yaml") for f in os.listdir(os.path.join(track_dir, d)))]
def get_tracks_from_file(file_path):
    """Read track paths from a text file (one per line)."""
    with open(file_path, "r") as f:
        tracks = [line.strip() for line in f.readlines() if line.strip()]
    return tracks

def compute_curvature(x, y):
        """
        Compute curvature κ for a 2D curve parameterized by x(s), y(s).
        Uses central finite differences for stability.

        Inputs:
            x, y : arrays of shape (N,)

        Returns:
            curvature : array of shape (N,) with κ at each point
        """
        x = np.asarray(x)
        y = np.asarray(y)

        # First derivatives
        dx = np.gradient(x)
        dy = np.gradient(y)

        # Second derivatives
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Curvature formula
        curvature = np.abs(dx * ddy - dy * ddx) / (dx*dx + dy*dy)**1.5

        # Handle divide-by-zero (straight segments)
        curvature[np.isnan(curvature)] = 0.0
        curvature[np.isinf(curvature)] = 0.0

        return curvature