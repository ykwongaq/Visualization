"""Bounding box visualization package.

Provides tools for drawing axis-aligned bounding boxes on images with
configurable colors, line widths, and text labels.

Modules:
    visualizer: Draw bounding boxes on RGB images.

Typical usage::

    from bbox.visualizer import Visualizer, VisualizerConfig

    viz = Visualizer(line_width=2, default_color=(0, 255, 0))
    output = viz.draw_bounding_boxes(image, bboxes, labels=["cat", "dog"])
"""

from bbox.visualizer import Visualizer, VisualizerConfig

__all__ = ["Visualizer", "VisualizerConfig"]
