import argparse
import os
import subprocess


def frame_to_video(
    input_folder: str, output_video: str, fps: int = 30, frame_pattern="%08d.jpg"
) -> None:
    """
    Convert frames to video using ffmpeg

    Args:
        frame_pattern (str): Path pattern to input frames (e.g., /path/to/frames/%04d.jpg)
        output_video (str): Path to the output video file
        fps (int): Frames per second for the output video
    """

    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    target_frame_pattern = os.path.join(input_folder, frame_pattern)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),  # input frame rate
        "-start_number",
        "0",
        "-i",
        target_frame_pattern,  # input pattern of image frames
        # "-c:v",
        # "libx264",  # use H.264 codec
        "-pix_fmt",
        "yuv420p",  # ensure compatibility with most players
        output_video,
    ]

    subprocess.run(cmd)


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
    frame_to_video(args.input_pattern, args.output_video, args.fps)
