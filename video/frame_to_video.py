import os
import argparse
import subprocess


def main(args):
    input_pattern = args.input_pattern
    output_video = args.output_video
    fps = args.fps

    cmd = [
        "ffmpeg",
        "-framerate",
        str(fps),  # input frame rate
        "-start_number",
        "0",
        "-i",
        input_pattern,  # input pattern of image frames
        "-c:v",
        "libx264",  # use H.264 codec
        "-pix_fmt",
        "yuv420p",  # ensure compatibility with most players
        output_video,
    ]

    subprocess.run(cmd)

    print(f"Video saved to {output_video}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_pattern", type=str, help="Path to the folder containing video frames"
    )
    parser.add_argument(
        "--output_video", type=str, help="Path to the output video file"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second for the output video"
    )

    args = parser.parse_args()
    main(args)
