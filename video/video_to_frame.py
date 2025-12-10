import os
import argparse
import subprocess

def get_fps(video_path: str) -> float:
    command = [
        "ffprobe",
        "-v", "0",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    fps_str = result.stdout.strip()
    
    # Handle fractional FPS like 30000/1001
    if "/" in fps_str:
        num, denom = map(float, fps_str.split("/"))
        fps = num / denom
    else:
        fps = float(fps_str)
    
    return fps

def video_to_frame(input_video: str, output_dir: str, output_pattern = "%08d.jpg", fps: int = None) -> None:
    """
    Convert video to frames using ffmpeg

    Args:
        input_video (str): Path to the input video file
        output_pattern (str): Path pattern for the output frames (e.g., /path/to/frames/%04d.jpg)
        fps (int): Frames per second to extract from the video
    """

    if fps is None:
        fps = get_fps(input_video)

    frame_pattern = os.path.join(output_dir, output_pattern)
    
    cmd = [
        "ffmpeg",
        "-i",
        input_video,  # input video file
        "-vf",
        f"fps={fps}",  # set frame rate
        "-start_number",
        "0",
        frame_pattern,  # output pattern for image frames
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
