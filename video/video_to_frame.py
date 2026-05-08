import argparse
import os
import subprocess


def get_fps(video_path: str) -> float:
    """
    Extract FPS from video file using ffprobe.

    Args:
        video_path: Path to the video file

    Returns:
        float: Frame rate (FPS) of the video

    Raises:
        RuntimeError: If ffprobe command fails
        ValueError: If no valid frame rate is found
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]

    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    # Check if ffprobe command succeeded
    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        raise RuntimeError(f"ffprobe failed for video '{video_path}': {error_msg}")

    fps_str = result.stdout.strip()

    # Check if output is empty (no video stream found)
    if not fps_str:
        raise ValueError(
            f"No video stream or frame rate data found for '{video_path}'. "
            f"The file might be corrupted, audio-only, or not a valid video file."
        )

    # Parse the frame rate
    try:
        if "/" in fps_str:
            num, denom = map(float, fps_str.split("/"))

            # Check for division by zero or invalid values
            if denom == 0:
                raise ValueError(
                    f"Invalid frame rate '{fps_str}' (division by zero) "
                    f"for video '{video_path}'"
                )

            if num == 0:
                raise ValueError(
                    f"Invalid frame rate '{fps_str}' (zero numerator) "
                    f"for video '{video_path}'"
                )

            fps = num / denom
        else:
            fps = float(fps_str)

        # Validate final FPS value
        if fps <= 0:
            raise ValueError(f"Invalid FPS value {fps} for video '{video_path}'")

        # Optional: check for unrealistic FPS values
        if fps > 1000:
            raise ValueError(
                f"Unrealistic FPS value {fps} for video '{video_path}'. "
                f"Raw value: {fps_str}"
            )

        return fps

    except ValueError as e:
        # Re-raise our own ValueError with context
        if "Invalid" in str(e) or "Unrealistic" in str(e):
            raise
        # Handle parsing errors
        raise ValueError(
            f"Failed to parse frame rate '{fps_str}' for video '{video_path}': {e}"
        )


def get_codec(video_path: str) -> str:
    """Return codec name of the video (e.g., h264, hevc, prores, vp9)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    codec = result.stdout.strip()
    return codec


def get_resolution(video_path: str) -> tuple[int, int]:
    """Return (width, height) of the input video."""
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        video_path,
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    w, h = map(int, result.stdout.strip().split("x")[:2])
    return w, h


def video_to_frame(
    input_video: str,
    output_dir: str,
    output_pattern: str = "%08d.jpg",
    fps: int = None,
    max_size: int = 1920,
    codec: str = None,
) -> None:
    """
    Convert video to frames using ffmpeg.
    Only scales down if the input video is larger than max_size.
    """

    if fps is None:
        fps = get_fps(input_video)

    if codec is None:
        codec = get_codec(input_video)

    os.makedirs(output_dir, exist_ok=True)
    frame_pattern = os.path.join(output_dir, output_pattern)

    # Check resolution before deciding to scale
    width, height = get_resolution(input_video)
    longer_side = max(width, height)

    if longer_side > max_size:
        # Add scaling to the filter
        scale_filter = (
            f"scale={max_size}:{max_size}:force_original_aspect_ratio=decrease"
        )
        vf_filter = f"fps={fps},{scale_filter}"
        print(f"Scaling down: {width}x{height} → max {max_size}")
    else:
        # No scaling needed
        vf_filter = f"fps={fps}"
        print(f"Keeping original size: {width}x{height}")

    # Codec-specific safety options
    codec_safety_flags = []
    if codec in ("h264", "hevc"):
        # Inter-frame codecs → ensure frames decode sequentially
        codec_safety_flags = [
            "-err_detect",
            "ignore_err",
            "-fflags",
            "+genpts",
            "-threads",
            "1",
        ]
    elif codec in ("mjpeg", "prores", "vp9", "av1"):
        # Less problem-prone codecs
        codec_safety_flags = ["-fflags", "+genpts"]

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        *codec_safety_flags,
        "-i",
        input_video,
        "-vf",
        vf_filter,
        "-start_number",
        "0",
        "-q:v",
        "2",
        frame_pattern,
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
    parser.add_argument(
        "--input_video", type=str, required=True, help="Path to the input video file"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Path to the output directory"
    )
    parser.add_argument(
        "--output_pattern", type=str, default="%08d.jpg", help="Output filename pattern"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frames per second to extract (default: video FPS)",
    )
    parser.add_argument(
        "--max_size", type=int, default=1920, help="Maximum pixel size for longest side"
    )
    parser.add_argument(
        "--codec", type=str, default=None, help="Codec of the input video"
    )

    args = parser.parse_args()
    main(args)
