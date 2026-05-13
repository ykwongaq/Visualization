"""Depth map visualization utilities.

This module provides :class:`Visualizer` for converting 2-D floating-point
depth maps into false-color RGB images using any matplotlib colormap.
Invalid pixels (NaN, Inf, or non-positive values) are masked out and
rendered as black.

Typical usage::

    from depth.visualizer import Visualizer, VisualizerConfig

    config = VisualizerConfig(color_map="plasma", reverse=True)
    viz = Visualizer(config)

    rgb = viz.visualize_depth(depth_array)
"""

import argparse

import matplotlib
import numpy as np

from dataclasses import dataclass
from typing import Optional


@dataclass
class VisualizerConfig:
    """Tunable rendering parameters for :class:`Visualizer`.

    All fields have sensible defaults so you only need to specify the ones you
    want to change.

    Attributes:
        color_map: Name of any matplotlib colormap used to map normalized
            depth values to colors (e.g. ``"viridis"``, ``"plasma"``,
            ``"inferno"``, ``"magma"``, ``"turbo"``).
            Defaults to ``"viridis"``.
        reverse: When ``True``, inverts the normalized depth before applying
            the colormap, so that closer surfaces appear brighter instead of
            darker (or vice-versa depending on the chosen colormap).
            Defaults to ``False``.

    Example::

        # Bright-near coloring with a warm colormap
        config = VisualizerConfig(color_map="plasma", reverse=True)
        viz = Visualizer(config)

        # Tweak a parameter after construction
        viz.config.reverse = False
    """

    color_map: str = "viridis"
    reverse: bool = False


class Visualizer:
    """Converts depth maps to false-color RGB images.

    Normalization is performed using only valid pixels (finite, positive
    values) so that NaN / Inf outliers do not collapse the visible range.
    Invalid pixels are always rendered as black regardless of the colormap.

    All methods return a new array and leave the input depth map unchanged.

    Example::

        import numpy as np
        import cv2
        from depth.visualizer import Visualizer, VisualizerConfig

        depth = np.load("depth.npy")

        viz = Visualizer()
        rgb = viz.visualize_depth(depth)

        cv2.imwrite("depth_color.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
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
                ``color_map="plasma"`` or ``reverse=True``. Ignored when
                ``config`` is supplied.

        Raises:
            ValueError: If both ``config`` and ``kwargs`` are provided.
            TypeError: If any kwarg does not match a :class:`VisualizerConfig`
                field (raised by the dataclass constructor).

        Example::

            # Default settings (viridis colormap, no reversal)
            viz = Visualizer()

            # Via keyword arguments
            viz = Visualizer(color_map="turbo", reverse=True)

            # Via a config object (useful when sharing config across instances)
            config = VisualizerConfig(color_map="plasma")
            viz = Visualizer(config)
        """
        if config is not None and kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")
        self.config = config if config is not None else VisualizerConfig(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def visualize_depth(
        self,
        depth: np.ndarray,
        reverse: Optional[bool] = None,
    ) -> np.ndarray:
        """Convert a 2-D depth map to a false-color RGB image.

        The valid-pixel range is normalized to ``[0, 1]`` before the colormap
        is applied, so the full color range is always used regardless of the
        absolute depth scale.

        Args:
            depth: 2-D array of shape ``(H, W)`` containing depth values.
                Any dtype is accepted; NaN, Inf, and non-positive values are
                treated as invalid and rendered as black.
            reverse: When ``True``, inverts the normalized depth so that
                closer surfaces appear brighter. When ``None`` (default),
                falls back to :attr:`VisualizerConfig.reverse`.

        Returns:
            RGB image of shape ``(H, W, 3)`` and dtype ``uint8``.
            Invalid pixels are set to ``(0, 0, 0)``.

        Raises:
            ValueError: If ``depth`` is not a 2-D array.

        Example::

            # Default colormap and direction from config
            rgb = viz.visualize_depth(depth)

            # Override reversal for this call only
            rgb = viz.visualize_depth(depth, reverse=True)
        """
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")

        if reverse is None:
            reverse = self.config.reverse

        valid_mask = np.isfinite(depth) & (depth > 0)

        if not np.any(valid_mask):
            h, w = depth.shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        d_min = depth[valid_mask].min()
        d_max = depth[valid_mask].max()

        normalized = (depth - d_min) / (d_max - d_min + 1e-8)
        normalized = np.clip(normalized, 0.0, 1.0)

        if reverse:
            normalized = 1.0 - normalized

        cmap = matplotlib.colormaps[self.config.color_map]
        rgb = (cmap(normalized)[..., :3] * 255).astype(np.uint8)
        rgb[~valid_mask] = 0

        return rgb


def main(args):
    depth = np.load(args.depth_path)

    viz = Visualizer()
    rgb = viz.visualize_depth(depth)

    import cv2
    cv2.imwrite(args.output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    main(parser.parse_args())
