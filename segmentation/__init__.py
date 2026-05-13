"""Segmentation utilities package.

Provides tools for visualizing and cropping segmented regions from images.

Modules:
    visualizer: Overlay binary and semantic segmentation masks on images.
    cropper:    Crop a masked region from an image and export it as RGBA.

Typical usage::

    from segmentation.visualizer import Visualizer, VisualizerConfig
    from segmentation.cropper import crop_image
"""

from segmentation.visualizer import Visualizer, VisualizerConfig
from segmentation.cropper import crop_image

__all__ = ["Visualizer", "VisualizerConfig", "crop_image"]
