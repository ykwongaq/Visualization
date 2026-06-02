"""CLI script for reporting the frame rate of a video file.

Typical usage::

    python -m video.check_fps --video_path clip.mp4
"""

import argparse

from yk_visualization.video.video_to_frames import get_fps


def main(args):
    fps = get_fps(args.video_path)
    print(f"Video path: {args.video_path}")
    print(f"FPS: {fps}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print the frame rate of a video file."
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to the video file."
    )
    main(parser.parse_args())
