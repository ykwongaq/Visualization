"""Segmentation mask visualization utilities.

This module provides :class:`Visualizer` for overlaying binary and semantic
segmentation masks on images, with configurable colors, opacity, and boundary
rendering.

Typical usage::

    from segmentation.visualizer import Visualizer, VisualizerConfig

    config = VisualizerConfig(mask_alpha=0.5, boundary_ratio=0.015)
    viz = Visualizer(config)

    output = viz.visualize_masks(image, masks, category_ids=[1, 3])
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union

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

Contains 30 visually distinct pastel/mid-tone colors. When the number of
categories exceeds 30 the index wraps around (``category_id % 30``).
"""


@dataclass
class VisualizerConfig:
    """Tunable rendering parameters for :class:`Visualizer`.

    All fields have sensible defaults so you only need to specify the ones you
    want to change.

    Attributes:
        mask_alpha: Opacity of the filled mask overlay in ``[0, 1]``.
            ``0`` = fully transparent (mask invisible),
            ``1`` = fully opaque (original image hidden).
            Defaults to ``0.4``.
        boundary_alpha: Opacity of the boundary stroke in ``[0, 1]``.
            Higher values make the boundary stand out more against the image.
            Defaults to ``0.7``.
        boundary_ratio: Boundary stroke width expressed as a fraction of the
            image's shorter side. For example, ``0.01`` on a 1080-pixel tall
            image gives a ~10-pixel-wide boundary. Clamped to a minimum of
            1 pixel at render time. Defaults to ``0.01``.
        color_map: Mapping from integer category index to ``[R, G, B]`` color.
            Defaults to the built-in :data:`COLOR_MAPS` palette (30 colors).
            Provide a custom dict to override colors for specific categories.

    Example::

        # Increase mask opacity and use a thicker boundary
        config = VisualizerConfig(mask_alpha=0.6, boundary_ratio=0.02)
        viz = Visualizer(config)

        # Tweak a single parameter after construction
        viz.config.boundary_alpha = 0.9

        # Swap in a custom color map
        viz.config.color_map = {0: [255, 0, 0], 1: [0, 255, 0]}
    """

    mask_alpha: float = 0.4
    boundary_alpha: float = 0.7
    boundary_ratio: float = 0.01
    color_map: Dict[int, List[int]] = field(default_factory=lambda: dict(COLOR_MAPS))


class Visualizer:
    """Overlays segmentation masks on images with configurable styling.

    Supports three visualization modes:

    * **Instance masks** — one binary mask per object
      (:meth:`visualize_masks`).
    * **Semantic mask** — a single integer-labeled mask where each value is a
      class index (:meth:`visualize_semantic_mask`).
    * **Boundary only** — draw just the contour of a mask
      (:meth:`visualize_boundary`).

    All methods return a new array and leave the input image unchanged.

    Example::

        import cv2
        from segmentation.visualizer import Visualizer, VisualizerConfig

        image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

        viz    = Visualizer(mask_alpha=0.5)
        output = viz.visualize_masks(image, mask, category_ids=2)

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
                ``mask_alpha=0.6`` or ``boundary_ratio=0.02``. Ignored when
                ``config`` is supplied.

        Raises:
            ValueError: If both ``config`` and ``kwargs`` are provided.
            TypeError: If any kwarg does not match a :class:`VisualizerConfig`
                field (raised by the dataclass constructor).

        Example::

            # Default settings
            viz = Visualizer()

            # Via keyword arguments
            viz = Visualizer(mask_alpha=0.6, boundary_ratio=0.02)

            # Via a config object (useful when sharing config across instances)
            config = VisualizerConfig(mask_alpha=0.6)
            viz = Visualizer(config)
        """
        if config is not None and kwargs:
            raise ValueError("Provide either config or keyword arguments, not both.")
        self.config = config if config is not None else VisualizerConfig(**kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def visualize_masks(
        self,
        image: np.ndarray,
        masks: Union[np.ndarray, List[np.ndarray]],
        category_ids: Union[int, List[int]] = 0,
        colors: Optional[
            Union[Tuple[int, int, int], List[Tuple[int, int, int]]]
        ] = None,
        add_boundary: bool = True,
    ) -> np.ndarray:
        """Overlay one or more binary masks on an image.

        Each mask is blended with a solid color at :attr:`VisualizerConfig.mask_alpha`
        opacity. An optional boundary stroke is drawn on top using
        :attr:`VisualizerConfig.boundary_alpha`.

        Args:
            image: RGB image of shape ``(H, W, C)`` and dtype ``uint8``.
            masks: A single binary mask of shape ``(H, W)``, or a list of
                such masks. Non-zero pixels are treated as foreground.
            category_ids: Integer category index for each mask, used to look up
                a color from :attr:`VisualizerConfig.color_map`. Supply a single
                ``int`` (applied to all masks) or a list with one entry per
                mask. Indices wrap around when they exceed the palette size.
                Defaults to ``0``.
            colors: Explicit ``(R, G, B)`` color or list of colors, one per
                mask. Overrides the color map when provided. A single tuple is
                broadcast to all masks.
            add_boundary: When ``True`` (default), calls :meth:`visualize_boundary`
                for each mask after blending the fill.

        Returns:
            A copy of ``image`` with every mask overlaid. The original array
            is never modified.

        Raises:
            AssertionError: If ``image`` is not 3-D, any mask is not 2-D, the
                spatial dimensions of image and a mask do not match, or the
                lengths of ``masks``, ``category_ids``, and ``colors`` differ.

        Example::

            # Single mask, default color from category 1
            output = viz.visualize_masks(image, mask, category_ids=1)

            # Multiple masks with explicit colors
            output = viz.visualize_masks(
                image,
                masks=[mask_a, mask_b],
                category_ids=[0, 1],
                colors=[(255, 0, 0), (0, 255, 0)],
            )

            # Disable boundary strokes
            output = viz.visualize_masks(image, mask, add_boundary=False)
        """
        assert image.ndim == 3, f"Expected 3D image (H, W, C), got {image.ndim}D"

        if isinstance(masks, np.ndarray):
            masks = [masks]
        if isinstance(category_ids, int):
            category_ids = [category_ids] * len(masks)

        n_colors = len(self.config.color_map)
        if colors is None:
            colors = [self.config.color_map[cid % n_colors] for cid in category_ids]
        elif isinstance(colors, tuple):
            colors = [colors] * len(masks)

        assert len(masks) == len(category_ids) == len(colors), (
            f"masks, category_ids, and colors must have equal length; "
            f"got {len(masks)}, {len(category_ids)}, {len(colors)}"
        )

        output = image.copy()
        for mask, category_id, color in zip(masks, category_ids, colors):
            assert mask.ndim == 2, f"Expected 2D mask (H, W), got {mask.ndim}D"
            assert image.shape[:2] == mask.shape, (
                f"Image and mask spatial dims must match: "
                f"{image.shape[:2]} vs {mask.shape}"
            )

            binary = (mask > 0).astype(np.uint8)
            color_layer = np.zeros_like(image)
            color_layer[binary == 1] = color

            blended = cv2.addWeighted(
                output, 1 - self.config.mask_alpha,
                color_layer, self.config.mask_alpha, 0,
            )
            output = np.where(binary[:, :, None] == 1, blended, output)

            if add_boundary:
                output = self.visualize_boundary(
                    output, binary, category_id=category_id, color=color
                )

        return output

    def visualize_boundary(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        category_id: int = 0,
        color: Optional[Tuple[int, int, int]] = None,
        boundary_width: Optional[int] = None,
    ) -> np.ndarray:
        """Draw a colored boundary stroke along the edge of a binary mask.

        The boundary is computed by subtracting an eroded version of the mask
        from itself, producing a thin ring that follows the mask contour. The
        ring is then blended with the image at
        :attr:`VisualizerConfig.boundary_alpha` opacity.

        Args:
            image: RGB image of shape ``(H, W, C)`` and dtype ``uint8``.
            mask: Binary mask of shape ``(H, W)``. Non-zero pixels are treated
                as foreground.
            category_id: Category index used to look up a color from
                :attr:`VisualizerConfig.color_map` when ``color`` is not given.
                Defaults to ``0``.
            color: Explicit ``(R, G, B)`` boundary color. Overrides the color
                map when provided.
            boundary_width: Stroke width in pixels. When ``None`` (default),
                the width is computed as
                ``max(1, int(min(H, W) * boundary_ratio))``.

        Returns:
            A copy of ``image`` with the boundary stroke overlaid. The
            original array is never modified.

        Raises:
            AssertionError: If ``image`` is not 3-D, ``mask`` is not 2-D, or
                their spatial dimensions do not match.

        Example::

            # Boundary with automatic width from config
            output = viz.visualize_boundary(image, mask, category_id=2)

            # Fixed 5-pixel boundary in red
            output = viz.visualize_boundary(
                image, mask, color=(255, 0, 0), boundary_width=5
            )
        """
        assert image.ndim == 3, f"Expected 3D image (H, W, C), got {image.ndim}D"
        assert mask.ndim == 2, f"Expected 2D mask (H, W), got {mask.ndim}D"
        assert image.shape[:2] == mask.shape, (
            f"Image and mask spatial dims must match: "
            f"{image.shape[:2]} vs {mask.shape}"
        )

        binary = (mask > 0).astype(np.uint8)
        h, w = image.shape[:2]

        if color is None:
            n_colors = len(self.config.color_map)
            color = self.config.color_map[category_id % n_colors]
        if boundary_width is None:
            boundary_width = max(1, int(min(h, w) * self.config.boundary_ratio))

        kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS, (boundary_width, boundary_width)
        )
        boundary = binary - cv2.erode(binary, kernel, iterations=1)

        boundary_layer = np.zeros_like(image, dtype=np.uint8)
        boundary_layer[boundary == 1] = color

        blended = cv2.addWeighted(
            image, 1 - self.config.boundary_alpha,
            boundary_layer, self.config.boundary_alpha, 0,
        )
        return np.where(boundary[:, :, None] == 1, blended, image)

    def visualize_semantic_mask(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        ignore_idx: Optional[Set[int]] = None,
    ) -> np.ndarray:
        """Visualize a semantic segmentation mask on an image.

        Unlike :meth:`visualize_masks`, which accepts separate binary masks,
        this method takes a single integer-labeled mask where each pixel value
        is a class index. Each unique class is colored independently and
        rendered via :meth:`visualize_masks`.

        Args:
            image: RGB image of shape ``(H, W, C)`` and dtype ``uint8``.
            mask: Semantic mask of shape ``(H, W)`` with integer class indices
                as pixel values (e.g. ``0 = background``, ``1 = cat``, …).
            ignore_idx: Set of class indices to skip entirely (not rendered).
                Defaults to ``{0}`` to suppress the background class. Pass an
                empty set ``set()`` to render all classes including background.

        Returns:
            A copy of ``image`` with each class region colored and bounded.
            The original array is never modified.

        Raises:
            AssertionError: If ``image`` is not 3-D, ``mask`` is not 2-D, or
                their spatial dimensions do not match.

        Example::

            # Default: skip class 0 (background)
            output = viz.visualize_semantic_mask(image, semantic_mask)

            # Render all classes including background
            output = viz.visualize_semantic_mask(
                image, semantic_mask, ignore_idx=set()
            )

            # Skip background (0) and a void/unlabeled class (255)
            output = viz.visualize_semantic_mask(
                image, semantic_mask, ignore_idx={0, 255}
            )
        """
        assert image.ndim == 3, f"Expected 3D image (H, W, C), got {image.ndim}D"
        assert mask.ndim == 2, f"Expected 2D mask (H, W), got {mask.ndim}D"
        assert image.shape[:2] == mask.shape, (
            f"Image and mask spatial dims must match: "
            f"{image.shape[:2]} vs {mask.shape}"
        )

        if ignore_idx is None:
            ignore_idx = {0}

        output = image.copy()
        for class_id in np.unique(mask):
            if class_id in ignore_idx:
                continue
            binary = (mask == class_id).astype(np.uint8)
            output = self.visualize_masks(output, binary, category_ids=int(class_id))
        return output
