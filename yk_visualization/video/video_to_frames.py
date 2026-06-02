"""Video-to-frame extraction utilities.

This module provides functions for extracting frames from a video file using
ffmpeg/ffprobe, with automatic codec detection, optional FPS override, and
safe downscaling for large inputs.

Typical usage::

    from video.video_to_frame import video_to_frame

    video_to_frame("input.mp4", "frames/", fps=24, max_size=1280)
"""

import argparse
import os
import subprocess
from typing import Optional


def get_fps(video_path: str) -> float:
    """Return the frame rate of a video file.

    Uses ``ffprobe`` to read the ``r_frame_rate`` field of the first video
    stream and converts the rational value (e.g. ``"30000/1001"``) to a float.

    Args:
        video_path: Path to the video file.

    Returns:
        Frame rate in frames per second.

    Raises:
        RuntimeError: If ``ffprobe`` exits with a non-zero return code.
        ValueError: If no video stream is found, the frame rate string cannot
            be parsed, or the resulting value is zero, negative, or
            unrealistically high (> 1000 fps).

    Example::

        fps = get_fps("clip.mp4")
        print(f"Frame rate: {fps:.3f} fps")
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

    if result.returncode != 0:
        error_msg = result.stderr.strip() or "Unknown error"
        raise RuntimeError(f"ffprobe failed for '{video_path}': {error_msg}")

    fps_str = result.stdout.strip()

    if not fps_str:
        raise ValueError(
            f"No video stream or frame rate data found for '{video_path}'. "
            "The file might be corrupted, audio-only, or not a valid video file."
        )

    try:
        if "/" in fps_str:
            num, denom = map(float, fps_str.split("/"))
            if denom == 0:
                raise ValueError(
                    f"Invalid frame rate '{fps_str}' (division by zero) "
                    f"for '{video_path}'"
                )
            if num == 0:
                raise ValueError(
                    f"Invalid frame rate '{fps_str}' (zero numerator) "
                    f"for '{video_path}'"
                )
            fps = num / denom
        else:
            fps = float(fps_str)

        if fps <= 0:
            raise ValueError(f"Invalid FPS value {fps} for '{video_path}'")
        if fps > 1000:
            raise ValueError(
                f"Unrealistic FPS value {fps} for '{video_path}'. "
                f"Raw value: {fps_str}"
            )

        return fps

    except ValueError as e:
        if "Invalid" in str(e) or "Unrealistic" in str(e):
            raise
        raise ValueError(
            f"Failed to parse frame rate '{fps_str}' for '{video_path}': {e}"
        )


def get_codec(video_path: str) -> str:
    """Return the codec name of the first video stream.

    Args:
        video_path: Path to the video file.

    Returns:
        Codec name as reported by ffprobe (e.g. ``"h264"``, ``"hevc"``,
        ``"prores"``, ``"vp9"``). Returns an empty string if no video stream
        is found.

    Example::

        codec = get_codec("clip.mp4")
        print(f"Codec: {codec}")
    """
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
    return result.stdout.strip()


def get_resolution(video_path: str) -> tuple[int, int]:
    """Return the pixel dimensions of the first video stream.

    Args:
        video_path: Path to the video file.

    Returns:
        ``(width, height)`` in pixels.

    Example::

        w, h = get_resolution("clip.mp4")
        print(f"Resolution: {w}x{h}")
    """
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


def video_to_frames(
    input_video: str,
    output_dir: str,
    output_pattern: str = "%08d.jpg",
    fps: Optional[float] = None,
    max_size: int = 1920,
    codec: Optional[str] = None,
) -> None:
    """Extract frames from a video file to a directory of images.

    Uses ffmpeg for extraction. If the video's longer side exceeds
    ``max_size``, frames are scaled down while preserving the aspect ratio.
    Codec-specific safety flags are applied automatically to reduce decoding
    errors on inter-frame codecs such as H.264 and HEVC.

    Args:
        input_video: Path to the source video file.
        output_dir: Directory where extracted frames are written. Created
            automatically if it does not exist.
        output_pattern: ``strftime``-style filename pattern for output frames,
            e.g. ``"%08d.jpg"`` (default) or ``"%04d.png"``.
        fps: Frame rate at which to sample the video. When ``None`` (default),
            the native FPS of the video is used (detected via :func:`get_fps`).
        max_size: Maximum allowed size in pixels for the longer side of each
            frame. Frames are scaled down proportionally when the source
            exceeds this value. Pass a large number (e.g. ``99999``) to
            disable scaling. Defaults to ``1920``.
        codec: Codec of the input video, used to select appropriate ffmpeg
            safety flags. When ``None`` (default), detected automatically via
            :func:`get_codec`.

    Raises:
        subprocess.CalledProcessError: If the ffmpeg command fails.
        RuntimeError: If FPS detection fails (propagated from :func:`get_fps`).

    Example::

        # Extract at native FPS, cap frames at 1280px on the longer side
        video_to_frame("input.mp4", "frames/", max_size=1280)

        # Extract at a fixed 1 fps (one frame per second)
        video_to_frame("input.mp4", "frames/", fps=1)
    """
    if fps is None:
        fps = get_fps(input_video)
    if codec is None:
        codec = get_codec(input_video)

    os.makedirs(output_dir, exist_ok=True)
    frame_pattern = os.path.join(output_dir, output_pattern)

    width, height = get_resolution(input_video)
    longer_side = max(width, height)

    if longer_side > max_size:
        scale_filter = (
            f"scale={max_size}:{max_size}:force_original_aspect_ratio=decrease"
        )
        vf_filter = f"fps={fps},{scale_filter}"
        print(f"Scaling down: {width}x{height} → max {max_size}")
    else:
        vf_filter = f"fps={fps}"
        print(f"Keeping original size: {width}x{height}")

    codec_safety_flags = []
    if codec in ("h264", "hevc"):
        codec_safety_flags = [
            "-err_detect",
            "ignore_err",
            "-fflags",
            "+genpts",
            "-threads",
            "1",
        ]
    elif codec in ("mjpeg", "prores", "vp9", "av1"):
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
    video_to_frames(
        args.input_video,
        args.output_dir,
        args.output_pattern,
        args.fps,
        args.max_size,
        args.codec,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from a video file using ffmpeg."
    )
    parser.add_argument(
        "--input_video",
        type=str,
        required=True,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write extracted frames.",
    )
    parser.add_argument(
        "--output_pattern",
        type=str,
        default="%08d.jpg",
        help="Output filename pattern (default: %%(08d).jpg).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Extraction frame rate (default: native video FPS).",
    )
    parser.add_argument(
        "--max_size",
        type=int,
        default=1920,
        help="Maximum pixels on the longer side (default: 1920).",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default=None,
        help="Input codec override (default: auto-detected).",
    )
    main(parser.parse_args())
