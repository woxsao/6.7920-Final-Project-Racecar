# train.py
import yaml
from argparse import Namespace
from stable_baselines3 import PPO
import numpy as np
import gym
from f110_gym.envs.base_classes import Integrator
from f1tenth_wrapper import F110SB3Wrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder


def train():
    # Load YAML config
    MAP_NAME = "0_straight"
    MAP_PATH = f"curricula/{MAP_NAME}"

    env = gym.make('f110_gym:f110-v0', map=MAP_PATH, map_ext='.png', num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    initial_pose = np.array([[0.7, 0.0, 0.0]], dtype=np.float32) # x, y, theta ?

    env = F110SB3Wrapper(env, start_pose=initial_pose)
    env.reset()

    env.render()

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="./tensorboard/f110_ppo/"  # <-- log directory
    )
    model.learn(total_timesteps=100_000)

    # Save the trained model
    model.save(f"{MAP_NAME}_ppo")
    
    
    
def test():
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

    # Load trained model
    model = PPO.load("f110_ppo_model", env=vec_env)

    obs = vec_env.reset()
    done = [False]

    while not done[0]:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)

        print(f"Obs: {obs}, Reward: {reward}, Done: {done}, Info: {info}")
        print()
        vec_env.render()  # Show simulation

    vec_env.close()



if __name__ == "__main__":
    print("Starting training...")
    train()
    
    print("Training complete. Starting testing...")
    test()