"""Frame-to-video assembly utilities.

This module provides :func:`frame_to_video` for assembling a directory of
sequentially numbered image frames into a video file using ffmpeg.

Typical usage::

    from video.frames_to_video import frames_to_video

    frames_to_video("frames/", "output.mp4", fps=30)
"""

import argparse
import os
import re
import subprocess


def _detect_start_number(input_folder: str, frame_pattern: str) -> int:
    """Find the smallest frame number matching ``frame_pattern`` in ``input_folder``.

    Converts a printf-style pattern (e.g. ``"%08d.jpg"``) to a regex and scans
    the directory for matches. Returns ``0`` if no matching files are found.

    Args:
        input_folder: Directory to scan for frame files.
        frame_pattern: ``printf``-style filename pattern (e.g. ``"%08d.jpg"``).

    Returns:
        The smallest integer found among matching filenames, or ``0``.
    """
    # Convert printf pattern to regex: %0Xd → (\d{X}), escape the rest
    regex_parts = []
    i = 0
    while i < len(frame_pattern):
        m = re.match(r"%0(\d+)d", frame_pattern[i:])
        if m:
            width = int(m.group(1))
            regex_parts.append(rf"(\d{{{width}}})")
            i += len(m.group(0))
        else:
            regex_parts.append(re.escape(frame_pattern[i]))
            i += 1
    regex = "^" + "".join(regex_parts) + "$"
    compiled = re.compile(regex)

    numbers: list[int] = []
    try:
        for fname in os.listdir(input_folder):
            match = compiled.match(fname)
            if match:
                numbers.append(int(match.group(1)))
    except FileNotFoundError:
        return 0

    return min(numbers) if numbers else 0


def frames_to_video(
    input_folder: str,
    output_video: str,
    fps: int = 30,
    frame_pattern: str = "%08d.jpg",
) -> None:
    """Assemble sequentially numbered image frames into a video file.

    Uses ffmpeg to read frames matching ``frame_pattern`` from
    ``input_folder`` and encode them into ``output_video``. The starting
    frame number is detected automatically by scanning the directory. The
    output container and codec are inferred from the ``output_video`` file
    extension. Pixel format is forced to ``yuv420p`` for broad player
    compatibility.

    Args:
        input_folder: Directory containing the input frames.
        output_video: Path to the output video file. The parent directory is
            created automatically if it does not exist. The file extension
            determines the container (e.g. ``".mp4"``, ``".avi"``, ``".mov"``).
        fps: Frame rate of the output video in frames per second.
            Defaults to ``30``.
        frame_pattern: ``strftime``-style filename pattern used to locate the
            input frames, e.g. ``"%08d.jpg"`` (default) or ``"%04d.png"``.
            Must match the naming produced by :func:`video.video_to_frame.video_to_frame`.

    Raises:
        subprocess.CalledProcessError: If the ffmpeg command fails.

    Example::

        # Assemble default-named frames at 24 fps
        frames_to_video("frames/", "output.mp4", fps=24)

        # Assemble PNG frames with a 4-digit pattern
        frames_to_video("frames/", "output.mov", fps=60, frame_pattern="%04d.png")
    """
    os.makedirs(os.path.dirname(output_video) or ".", exist_ok=True)

    start_number = _detect_start_number(input_folder, frame_pattern)
    target_frame_pattern = os.path.join(input_folder, frame_pattern)

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-start_number",
        str(start_number),
        "-i",
        target_frame_pattern,
        "-pix_fmt",
        "yuv420p",
        output_video,
    ]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Assemble image frames into a video file using ffmpeg."
    )
    parser.add_argument(
        "--frame_folder",
        type=str,
        required=True,
        help="Directory containing the input frames.",
    )
    parser.add_argument(
        "--output_video",
        type=str,
        required=True,
        help="Path to the output video file.",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="%08d.jpg",
        help="Input filename pattern (default: %%(08d).jpg).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output frame rate in fps (default: 30).",
    )

    args = parser.parse_args()
    frames_to_video(args.frame_folder, args.output_video, args.fps, args.file_pattern)
