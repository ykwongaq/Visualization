import os
import argparse
import subprocess


def video_to_frame(input_video: str, output_pattern: str, fps: int) -> None:
    """
    Convert video to frames using ffmpeg

    Args:
        input_video (str): Path to the input video file
        output_pattern (str): Path pattern for the output frames (e.g., /path/to/frames/%04d.jpg)
        fps (int): Frames per second to extract from the video
    """
    cmd = [
        "ffmpeg",
        "-i",
        input_video,  # input video file
        "-vf",
        f"fps={fps}",  # set frame rate
        "-start_number",
        "0",
        output_pattern,  # output pattern for image frames
    ]

    subprocess.run(cmd)

# Convert video to frames using ffmpeg
def main(args):
    input_video = args.input_video
    output_pattern = args.output_pattern
    fps = args.fps

    os.makedirs(os.path.dirname(output_pattern), exist_ok=True)

    video_to_frame(input_video, output_pattern, fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, help="Path to the input video file")
    parser.add_argument(
        "--output_pattern",
        type=str,
        help="Path pattern for the output frames (e.g., %04d.jpg)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second to extract from the video",
    )

    args = parser.parse_args()
    main(args)
