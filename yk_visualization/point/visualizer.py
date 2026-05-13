"""Point visualization utilities.

This module provides :class:`Visualizer` for drawing labeled points on images
as semi-transparent filled circles with a solid boundary stroke.

Typical usage::

    from point.visualizer import Visualizer, VisualizerConfig

    config = VisualizerConfig(radius_ratio=0.015, alpha=0.6)
    viz = Visualizer(config)

    output = viz.draw_points(image, points=[(120, 80), (300, 200)], labels=[0, 2])
"""

import numpy as np
import cv2

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

COLOR_MAPS: Dict[int, List[int]] = {
    0: [135, 206, 250],   # Light Sky Blue
    1: [255, 182, 193],   # Light Pink
    2: [152, 251, 152],   # Pale Green
    3: [255, 160, 122],   # Light Salmon
    4: [238, 130, 238],   # Violet
    5: [240, 230, 140],   # Khaki
    6: [250, 128, 114],   # Salmon
    7: [221, 160, 221],   # Plum
    8: [175, 238, 238],   # Pale Turquoise
    9: [255, 228, 181],   # Moccasin
    10: [144, 238, 144],  # Light Green
    11: [216, 191, 216],  # Thistle
    12: [255, 218, 185],  # Peach Puff
    13: [32, 178, 170],   # Light Sea Green
    14: [100, 149, 237],  # Cornflower Blue
    15: [244, 164, 96],   # Sandy Brown
    16: [72, 209, 204],   # Medium Turquoise
    17: [219, 112, 147],  # Pale Violet Red
    18: [255, 222, 173],  # Navajo White
    19: [255, 105, 180],  # Hot Pink
    20: [255, 127, 80],   # Coral
    21: [173, 216, 230],  # Light Blue
    22: [127, 255, 212],  # Aquamarine
    23: [240, 128, 128],  # Light Coral
    24: [250, 250, 210],  # Light Goldenrod Yellow
    25: [205, 92, 92],    # Indian Red
    26: [176, 224, 230],  # Powder Blue
    27: [210, 180, 140],  # Tan
    28: [255, 239, 213],  # Papaya Whip
    29: [222, 184, 135],  # Burlywood
}
"""Built-in color palette mapping category indices to ``[R, G, B]`` values.

Contains 30 visually distinct pastel/mid-tone colors. When a label exceeds
29 the index wraps around (``label % 30``).
"""


@dataclass
class VisualizerConfig:
    """Tunable rendering parameters for :class:`Visualizer`.

    All fields have sensible defaults so you only need to specify the ones you
    want to change.

    Attributes:
        radius_ratio: Radius of the filled circle expressed as a fraction of
            the image's shorter side. For example, ``0.01`` on a 500-pixel
            image gives a 5-pixel radius. Clamped to a minimum of 1 pixel at
            render time. Defaults to ``0.01``.
        boundary_width_ratio: Width of the solid boundary stroke expressed as
            a fraction of the image's shorter side. Clamped to a minimum of
            1 pixel at render time. Defaults to ``0.001``.
        alpha: Opacity of the filled circle interior in ``[0, 1]``.
            ``0`` = fully transparent, ``1`` = fully opaque.
            The boundary stroke is always drawn at full opacity.
            Defaults to ``0.5``.
        color_map: Mapping from integer label index to ``[R, G, B]`` color.
            Defaults to the built-in :data:`COLOR_MAPS` palette (30 colors).
            Provide a custom dict to override colors for specific labels.

    Example::

        # Larger, more opaque points
        config = VisualizerConfig(radius_ratio=0.02, alpha=0.8)
        viz = Visualizer(config)

        # Tweak a single parameter after construction
        viz.config.alpha = 0.6

        # Swap in a custom color map
        viz.config.color_map = {0: [255, 0, 0], 1: [0, 255, 0]}
    """

    radius_ratio: float = 0.01
    boundary_width_ratio: float = 0.001
    alpha: float = 0.5
    color_map: Dict[int, List[int]] = field(default_factory=lambda: dict(COLOR_MAPS))


class Visualizer:
    """Draws labeled points on images as alpha-blended filled circles.

    Each point is rendered in two layers:

    * A semi-transparent filled circle blended with
      :attr:`VisualizerConfig.alpha` opacity.
    * A fully opaque boundary stroke drawn on top for visibility.

    Both the circle radius and boundary width scale with the image size via
    ratio parameters so the result looks consistent across different
    resolutions.

    All methods return a new array and leave the input image unchanged.

    Example::

        import cv2
        from point.visualizer import Visualizer, VisualizerConfig

        image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

        viz = Visualizer()
        output = viz.draw_points(
            image,
            points=[(120, 80), (300, 200)],
            labels=[0, 2],
        )

        cv2.imwrite("output.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    """

    def __init__(
        self,
        config: Optional[VisualizerConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize the Visualizer.

        You can supply a pre-built :class:`VisualizerConfig` **or** pass
        individual config fields as keyword arguments — but not both.

        Args:
            config: A :class:`VisualizerConfig` instance that fully describes
                the rendering style. When provided, ``kwargs`` must be empty.
            **kwargs: Shorthand for any :class:`VisualizerConfig` field, e.g.
                ``radius_ratio=0.02`` or ``alpha=0.8``. Ignored when
                ``config`` is supplied.

        Raises:
            ValueError: If both ``config`` and ``kwargs`` are provided.
            TypeError: If any kwarg does not match a :class:`VisualizerConfig`
                field (raised by the dataclass constructor).

        Example::

            # Default settings
            viz = Visualizer()

            # Via keyword arguments
            viz = Visualizer(radius_ratio=0.015, alpha=0.7)

            # Via a config object (useful when sharing config across instances)
            config = VisualizerConfig(radius_ratio=0.02)
            viz = Visualizer(config)
        """
        if config is not None and kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")
        self.config = config if config is not None else VisualizerConfig(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_points(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int]],
        labels: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Draw labeled points on an image as alpha-blended filled circles.

        Each point is rendered as a filled circle whose interior is blended
        with the image at :attr:`VisualizerConfig.alpha` opacity. A fully
        opaque boundary stroke is drawn on top to ensure the point is visible
        against any background.

        Both the circle radius and boundary width are derived from
        :attr:`VisualizerConfig.radius_ratio` and
        :attr:`VisualizerConfig.boundary_width_ratio` respectively, scaled
        by the shorter side of the image.

        Args:
            image: RGB image of shape ``(H, W, C)`` and dtype ``uint8``.
            points: Pixel coordinates of the points to draw, each given as
                an ``(x, y)`` tuple. ``x`` is the horizontal axis (column)
                and ``y`` is the vertical axis (row).
            labels: Integer category index for each point, used to look up
                a color from :attr:`VisualizerConfig.color_map`. Supply one
                label per point; indices wrap around when they exceed the
                palette size. When ``None`` (default), all points are drawn
                in the color at index ``0``.

        Returns:
            A copy of ``image`` with every point drawn on it. The original
            array is never modified.

        Example::

            # All points in the default color
            output = viz.draw_points(image, [(50, 60), (200, 150)])

            # Per-point colors via labels
            output = viz.draw_points(
                image,
                points=[(50, 60), (200, 150)],
                labels=[0, 5],
            )
        """
        output = image.copy()
        h, w = output.shape[:2]
        min_side = min(h, w)
        radius = max(1, int(min_side * self.config.radius_ratio))
        boundary_width = max(1, int(min_side * self.config.boundary_width_ratio))

        overlay = np.zeros_like(output, dtype=np.uint8)
        n_colors = len(self.config.color_map)

        for i, (x, y) in enumerate(points):
            if labels is not None and i < len(labels):
                color = self.config.color_map.get(
                    int(labels[i]) % n_colors, COLOR_MAPS[0]
                )
            else:
                color = self.config.color_map.get(0, COLOR_MAPS[0])

            color_tuple = tuple(int(c) for c in color)
            cv2.circle(overlay, (int(x), int(y)), radius, color_tuple, -1)
            cv2.circle(output, (int(x), int(y)), radius, color_tuple, boundary_width)

        mask = overlay.any(axis=-1)
        blended = output.copy()
        blended[mask] = (
            (1 - self.config.alpha) * output[mask].astype(np.float32)
            + self.config.alpha * overlay[mask].astype(np.float32)
        ).astype(np.uint8)

        return blended
