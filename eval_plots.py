import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from utils import make_env_from_track, get_initial_pose

def run_and_plot(model_path, track_path, use_raceline=False, max_steps=2000):
    # --- Load trained model ---
    model = PPO.load(model_path)

    # --- Create environment using your training function ---
    initial_pose = get_initial_pose(track_path)
    env = make_env_from_track(track_path, use_raceline=use_raceline)
    obs = env.reset(poses=initial_pose)

    # --- Collect agent trajectory ---
    agent_xy_history = []

    done = False
    steps = 0
    MAX_STEPS = 200_000_000

    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, info = env.step(action)
        ego_idx = 0
        agent_xy_history.append([obs['poses_x'][ego_idx], obs['poses_y'][ego_idx]])
        # env.render(mode="human")
        steps += 1
    print("Final steps:", steps, "obs:",obs)
    env.close()

    agent_xy_history = np.array(agent_xy_history)

    # --- Plot trajectory vs raceline ---
    if hasattr(env, 'centerline_xy') and env.centerline_xy is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(env.centerline_xy[:, 0], env.centerline_xy[:, 1], label="Raceline", linewidth=2)
        plt.plot(agent_xy_history[:, 0], agent_xy_history[:, 1], label="Agent Path", linewidth=2)
        plt.plot(agent_xy_history[0, 0], agent_xy_history[0, 1], 'ro', label="Start")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Agent Trajectory vs Raceline")
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        plt.show()

    env.close()
    print(f"Episode finished in {steps} steps.")


if __name__ == "__main__":
    model_path = sys.argv[1]  # path to PPO model
    track_path = sys.argv[2]  # path to track folder
    run_and_plot(model_path, track_path)
