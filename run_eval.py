import subprocess

MODEL_PATH = "results/ppo_f110_multi_track_5_tracks_2mil_per_track"
VAL_TRACKS_FILE = "train_tracks.txt"   # or train_tracks.txt


def get_tracks_list(file_path):
    with open(file_path, "r") as f:
        return [line.strip() for line in f.readlines()]


def main():
    tracks = get_tracks_list(VAL_TRACKS_FILE)

    for i, track in enumerate(tracks):
        print("\n==============================")
        print(f"Running renderer for track {i+1}/{len(tracks)}: {track}")
        print("==============================")

        subprocess.run(
            [
                "python",
                "single_render.py",
                MODEL_PATH,
                track,
            ],
            check=True,
        )

    print("\nAll tracks finished!")


if __name__ == "__main__":
    main()
