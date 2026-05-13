"""Depth map visualization package.

Provides tools for converting floating-point depth maps into false-color
RGB images using matplotlib colormaps.

Modules:
    visualizer: Convert 2-D depth arrays to false-color RGB images.

Typical usage::

    from depth.visualizer import Visualizer, VisualizerConfig

    viz = Visualizer(color_map="plasma", reverse=True)
    rgb = viz.visualize_depth(depth_array)
"""

from depth.visualizer import Visualizer, VisualizerConfig

__all__ = ["Visualizer", "VisualizerConfig"]
