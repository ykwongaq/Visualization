"""Video utilities package.

Provides tools for extracting frames from videos and assembling frames back
into videos, along with helpers for reading video metadata.

Modules:
    video_to_frame: Extract frames from a video file using ffmpeg.
    frame_to_video: Assemble image frames into a video file using ffmpeg.

Typical usage::

    from video.video_to_frame import video_to_frame, get_fps
    from video.frame_to_video import frame_to_video

    video_to_frame("input.mp4", "frames/", fps=24)
    frame_to_video("frames/", "output.mp4", fps=24)
"""

from video.video_to_frame import get_codec, get_fps, get_resolution, video_to_frame
from video.frame_to_video import frame_to_video

__all__ = [
    "video_to_frame",
    "frame_to_video",
    "get_fps",
    "get_codec",
    "get_resolution",
]
