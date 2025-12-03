import sys
from stable_baselines3 import PPO, SAC
from utils import make_env_from_track, get_initial_pose


def run_one(model_path, track_path):
    model = PPO.load(model_path)

    initial_pose = get_initial_pose(track_path)
    env = make_env_from_track(track_path, use_raceline=False)
    obs = env.reset(poses=initial_pose)

    done = False
    steps = 0
    MAX_STEPS = 200_000_000

    while not done and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        print(action)
        obs, reward, done, info = env.step(action)
        env.render(mode="human")
        steps += 1
    print("Final steps:", steps, "obs:",obs)
    env.close()


if __name__ == "__main__":
    model_path = sys.argv[1]
    track_path = sys.argv[2]
    run_one(model_path, track_path)
