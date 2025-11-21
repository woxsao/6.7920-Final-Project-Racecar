# run.py
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from f1tenth_wrapper import F110SB3Wrapper
from f110_gym.envs.base_classes import Integrator
import yaml
from argparse import Namespace

# Load config
with open("envs/config_example_map.yaml") as f:
    conf_dict = yaml.safe_load(f)
conf = Namespace(**conf_dict)

# Create env
env = gym.make(
    'f110_gym:f110-v0',
    map=conf.map_path,
    map_ext=conf.map_ext,
    num_agents=1,
    timestep=0.01,
    integrator=Integrator.RK4
)
initial_pose = np.array([[conf.sx, conf.sy, conf.stheta]], dtype=np.float32)
env = F110SB3Wrapper(env, start_pose=initial_pose)

# Wrap in DummyVecEnv for SB3
vec_env = DummyVecEnv([lambda: env])

# Optional: record video
record = True
if record:
    vec_env = VecVideoRecorder(
        vec_env,
        video_folder="videos/",
        record_video_trigger=lambda x: True,
        video_length=1000
    )

# Load trained model
model = PPO.load("f110_ppo_model", env=vec_env)

obs = vec_env.reset()
done = [False]

while not done[0]:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()  # Show simulation

vec_env.close()
