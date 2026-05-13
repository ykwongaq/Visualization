"""Bounding box visualization utilities.

This module provides :class:`Visualizer` for drawing axis-aligned bounding
boxes on images, with configurable colors, line width, and optional text
labels.

Typical usage::

    from bbox.visualizer import Visualizer, VisualizerConfig

    config = VisualizerConfig(line_width=2, default_color=(0, 255, 0))
    viz = Visualizer(config)

    output = viz.draw_bounding_boxes(image, bboxes, labels=["cat", "dog"])
"""

import numpy as np

from dataclasses import dataclass
from PIL import Image, ImageDraw
from typing import List, Optional, Tuple, Union


@dataclass
class VisualizerConfig:
    """Tunable rendering parameters for :class:`Visualizer`.

    All fields have sensible defaults so you only need to specify the ones you
    want to change.

    Attributes:
        line_width: Stroke width in pixels for the bounding box outline.
            Defaults to ``3``.
        default_color: Fallback ``(R, G, B)`` color used when no explicit
            color is supplied to :meth:`Visualizer.draw_bounding_boxes`.
            Defaults to ``(255, 0, 0)`` (red).

    Example::

        # Thicker green boxes
        config = VisualizerConfig(line_width=5, default_color=(0, 255, 0))
        viz = Visualizer(config)

        # Tweak a single parameter after construction
        viz.config.line_width = 2
    """

    line_width: int = 3
    default_color: Tuple[int, int, int] = (255, 0, 0)


class Visualizer:
    """Draws axis-aligned bounding boxes on images.

    Supports drawing one or many boxes in a single call, with per-box color
    overrides and optional text labels. All methods return a new array and
    leave the input image unchanged.

    Example::

        import cv2
        from bbox.visualizer import Visualizer, VisualizerConfig

        image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)

        viz = Visualizer()
        output = viz.draw_bounding_boxes(
            image,
            bboxes=[(10, 20, 100, 80), (150, 50, 300, 200)],
            labels=["cat", "dog"],
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
                ``line_width=5`` or ``default_color=(0, 255, 0)``. Ignored
                when ``config`` is supplied.

        Raises:
            ValueError: If both ``config`` and ``kwargs`` are provided.
            TypeError: If any kwarg does not match a :class:`VisualizerConfig`
                field (raised by the dataclass constructor).

        Example::

            # Default settings
            viz = Visualizer()

            # Via keyword arguments
            viz = Visualizer(line_width=5, default_color=(0, 255, 0))

            # Via a config object (useful when sharing config across instances)
            config = VisualizerConfig(line_width=2)
            viz = Visualizer(config)
        """
        if config is not None and kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")
        self.config = config if config is not None else VisualizerConfig(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        bboxes: Union[
            List[Tuple[int, int, int, int]], List[int], Tuple[int, int, int, int]
        ],
        colors: Optional[
            Union[Tuple[int, int, int], List[Tuple[int, int, int]]]
        ] = None,
        labels: Optional[List[str]] = None,
        line_width: Optional[int] = None,
    ) -> np.ndarray:
        """Draw one or more bounding boxes on an image.

        Each box is rendered as a colored rectangle outline. An optional text
        label is drawn at the top-left corner of each box.

        Args:
            image: RGB image of shape ``(H, W, C)`` and dtype ``uint8``.
            bboxes: Bounding box(es) to draw. Accepted formats:

                * A single box as a tuple: ``(x1, y1, x2, y2)``
                * A single box as a flat list: ``[x1, y1, x2, y2]``
                * Multiple boxes as a list of tuples/lists:
                  ``[(x1, y1, x2, y2), ...]``

                Coordinates are in pixel space with the origin at the
                top-left corner; ``x1 < x2`` and ``y1 < y2``.
            colors: Box outline color(s) as ``(R, G, B)`` tuples in
                ``[0, 255]``. Supply a single tuple to apply the same color
                to every box, or a list with one entry per box. When
                ``None`` (default), :attr:`VisualizerConfig.default_color`
                is used for all boxes.
            labels: Optional text labels, one per box. Each label is drawn
                at the top-left corner of its corresponding box using the
                same color as the outline. Defaults to ``None`` (no labels).
            line_width: Stroke width in pixels for the box outline. When
                ``None`` (default), :attr:`VisualizerConfig.line_width` is
                used.

        Returns:
            A copy of ``image`` with every bounding box (and label) drawn on
            it. The original array is never modified.

        Raises:
            ValueError: If ``bboxes`` is not one of the supported formats
                (raised by :meth:`normalize_bboxes`).
            AssertionError: If the number of boxes does not match the number
                of colors, or the number of boxes does not match the number
                of labels.

        Example::

            # Single red box, default line width
            output = viz.draw_bounding_boxes(image, (10, 20, 100, 80))

            # Multiple boxes with per-box colors and labels
            output = viz.draw_bounding_boxes(
                image,
                bboxes=[(10, 20, 100, 80), (150, 50, 300, 200)],
                colors=[(255, 0, 0), (0, 255, 0)],
                labels=["cat", "dog"],
            )

            # Override line width for this call only
            output = viz.draw_bounding_boxes(image, bbox, line_width=1)
        """
        bboxes = self.normalize_bboxes(bboxes)

        if len(bboxes) == 0:
            return image.copy()

        if colors is None:
            colors = [self.config.default_color] * len(bboxes)
        elif isinstance(colors, tuple):
            colors = [colors] * len(bboxes)

        if line_width is None:
            line_width = self.config.line_width

        assert len(bboxes) == len(colors), (
            f"Number of bounding boxes must match number of colors; "
            f"got {len(bboxes)} and {len(colors)}"
        )
        if labels is not None:
            assert len(bboxes) == len(labels), (
                f"Number of bounding boxes must match number of labels; "
                f"got {len(bboxes)} and {len(labels)}"
            )

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            color = tuple(colors[i])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            if labels is not None:
                draw.text((x1, y1), labels[i], fill=color)

        return np.array(pil_image)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def normalize_bboxes(
        self,
        bboxes: Union[
            List[Tuple[int, int, int, int]], List[int], Tuple[int, int, int, int]
        ],
    ) -> List[Tuple[int, int, int, int]]:
        """Normalize bounding boxes into a uniform list of ``(x1, y1, x2, y2)`` tuples.

        This helper is called internally by :meth:`draw_bounding_boxes` but can
        also be used directly when you need a consistent representation before
        further processing.

        Args:
            bboxes: Bounding box(es) in any of the following formats:

                * A single box as a tuple: ``(x1, y1, x2, y2)``
                * A single box as a flat list of four ints:
                  ``[x1, y1, x2, y2]``
                * Multiple boxes as a list of tuples/lists:
                  ``[(x1, y1, x2, y2), ...]``

        Returns:
            A list of ``(x1, y1, x2, y2)`` tuples, one per bounding box.

        Raises:
            ValueError: If the input does not match any supported format.

        Example::

            viz.normalize_bboxes((10, 20, 100, 80))
            # → [(10, 20, 100, 80)]

            viz.normalize_bboxes([10, 20, 100, 80])
            # → [(10, 20, 100, 80)]

            viz.normalize_bboxes([(10, 20, 100, 80), (5, 5, 50, 50)])
            # → [(10, 20, 100, 80), (5, 5, 50, 50)]
        """
        if isinstance(bboxes, tuple):
            return [bboxes]

        if isinstance(bboxes, list):
            if all(isinstance(b, (list, tuple)) for b in bboxes):
                return [tuple(b) for b in bboxes]
            if all(isinstance(b, int) for b in bboxes) and len(bboxes) == 4:
                return [tuple(bboxes)]

        raise ValueError(
            "Invalid bounding box format. Expected a tuple (x1, y1, x2, y2), "
            "a list of such tuples, or a flat list of 4 integers."
        )
