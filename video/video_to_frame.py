import argparse
import os
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


def video_to_frame(
    input_video: str,
    output_dir: str,
    output_pattern: str = "%08d.jpg",
    fps: int = None,
    max_size: int = 1024
) -> None:
    """
    Convert video to frames using ffmpeg.

    Args:
        input_video (str): Path to the input video file
        output_dir (str): Directory for output frames
        output_pattern (str): Filename pattern for the output frames (e.g., %04d.jpg)
        fps (int): Frames per second to extract from the video
        max_size (int): Maximum pixel size for the longer side of output frames
    """

    if fps is None:
        fps = get_fps(input_video)

    os.makedirs(output_dir, exist_ok=True)
    frame_pattern = os.path.join(output_dir, output_pattern)

    # scaling + fps filter, maintaining aspect ratio
    scale_filter = f"scale='if(gt(iw,ih),{max_size},-1)':'if(gt(ih,iw),{max_size},-1)'"
    vf_filter = f"fps={fps},{scale_filter}"

    cmd = [
        "ffmpeg",
        "-i", input_video,      # input video
        "-vf", vf_filter,       # video filter chain
        "-start_number", "0",
        frame_pattern           # output filename pattern
    ]

    subprocess.run(cmd, check=True)


def main(args):
    input_video = args.input_video
    output_dir = args.output_dir
    output_pattern = args.output_pattern
    fps = args.fps
    max_size = args.max_size

    video_to_frame(input_video, output_dir, output_pattern, fps, max_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory")
    parser.add_argument(
        "--output_pattern",
        type=str,
        default="%08d.jpg",
        help="Filename pattern for the output frames (default: %08d.jpg)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second to extract from the video",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1024,
        help="Maximum pixel size for the longer dimension of the frames",
    )

    args = parser.parse_args()
    main(args)