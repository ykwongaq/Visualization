import argparse
import subprocess

from video_to_frame import get_fps


def main(args):
    video_path = args.video_path

    fps = get_fps(video_path)
    print(f"Video path: {video_path}")
    print(f"FPS: {fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)

    args = parser.parse_args()
    main(args)
