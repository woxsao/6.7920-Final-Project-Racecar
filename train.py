# train.py
import yaml
from argparse import Namespace
from stable_baselines3 import PPO
import numpy as np
import gym
from f110_gym.envs.base_classes import Integrator
from f1tenth_wrapper import F110SB3Wrapper


# Load YAML config
with open("envs/config_example_map.yaml") as f:
    conf_dict = yaml.safe_load(f)
conf = Namespace(**conf_dict)

env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
initial_pose = np.array([[conf.sx, conf.sy, conf.stheta]])

initial_pose = np.array([[conf.sx, conf.sy, conf.stheta]], dtype=np.float32)
env = F110SB3Wrapper(env, start_pose=initial_pose)
env.reset()

env.render()

model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard/f110_ppo/"  # <-- log directory
)
model.learn(total_timesteps=1000_000)

# Save the trained model
model.save("f110_ppo_model")