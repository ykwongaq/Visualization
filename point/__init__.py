"""Point visualization package.

Provides tools for drawing labeled points on images as semi-transparent
filled circles with a solid boundary stroke.

Modules:
    visualizer: Draw points on RGB images with configurable size and opacity.

Typical usage::

    from point.visualizer import Visualizer, VisualizerConfig

    viz = Visualizer(radius_ratio=0.015, alpha=0.6)
    output = viz.draw_points(image, points=[(120, 80), (300, 200)], labels=[0, 2])
"""

from point.visualizer import Visualizer, VisualizerConfig

__all__ = ["Visualizer", "VisualizerConfig"]
