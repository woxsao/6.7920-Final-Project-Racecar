# run.py
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from f1tenth_wrapper import F110SB3Wrapper
from f110_gym.envs.base_classes import Integrator
import yaml
from argparse import Namespace
import matplotlib.pyplot as plt

from stable_baselines3.common.env_util import make_vec_env
plt.ion()

def render_scan(scan):
    plt.clf()
    plt.plot(scan)
    plt.title("LiDAR Scan")
    plt.pause(0.001)

MAX_STEPS = 100_000_000
# Load config
def load_config(path: str):
    with open(path, "r") as f:
        conf_dict = yaml.safe_load(f)
    conf = Namespace(**conf_dict)
    return conf

def run_eval(model_path, conf_path, initial_pose=None, env = None):
    conf = load_config(conf_path)
    # Create env
    if not env:
        env = gym.make(
            'f110_gym:f110-v0',
            map=conf.map_path,
            map_ext=conf.map_ext,
            num_agents=1,
            timestep=0.01,
            integrator=Integrator.RK4
        )
        if initial_pose is None:
            initial_pose = np.array([[conf.sx, conf.sy, conf.stheta]], dtype=np.float32)
        env = F110SB3Wrapper(env, start_pose=initial_pose)

    vec_env = DummyVecEnv([lambda: env])
    # vec_env = make_vec_env(lambda: env, n_envs=8)

    # Load trained model
    model = PPO.load(model_path, env=vec_env)

    obs = vec_env.reset()
    done = [False]

    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        vec_env.render()  # Show simulation
        render_scan(obs["scans"].flatten())
        

        

    vec_env.close()
if __name__ == "__main__":
    run_eval(model_path="f110_ppo_model", conf_path="envs/config_example_map.yaml")