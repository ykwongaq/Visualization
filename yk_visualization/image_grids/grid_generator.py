"""Image grid generation utilities.

This module provides :class:`GridGenerator` for composing multiple images into
a uniform grid layout with configurable padding and per-image text labels.

Typical usage::

    from yk_visualization.image_grids.grid_generator import GridGenerator, GridConfig

    config = GridConfig(padding=20, max_resolution=1920)
    gen = GridGenerator(config)

    # 2 × 2 grid
    gen.add_image(img1, label="A").add_image(img2, label="B")
    gen.add_image(img3, label="C").add_image(img4, label="D")
    output = gen.generate(rows=2, cols=2)

    # 1 × 3 grid (reuse the same instance after clearing)
    gen.clear()
    gen.add_image(img_a, "Input").add_image(img_b, "Prediction").add_image(img_c, "GT")
    output = gen.generate(cols=3)
"""

from __future__ import annotations

import cv2
import numpy as np

from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GridConfig:
    """Tunable rendering parameters for :class:`GridGenerator`.

    All fields have sensible defaults so you only need to specify the ones you
    want to change.

    Attributes:
        padding: Gap in pixels between adjacent cells (and, when
            *outer_padding* is ``True``, between cells and the outer
            border).  A single value applies to both axes.  Defaults to
            ``20``.
        outer_padding: When ``True`` (the default), *padding* is also
            applied to the outer border of the grid.  Set to ``False`` to
            keep the image content flush with the grid edges, with padding
            only between cells.
        max_resolution: Maximum width **or** height (in pixels) of the
            output grid.  The grid is sized so neither dimension exceeds
            this value while the cell dimensions remain uniform.  Defaults
            to ``1920``.
        tight: When ``True``, cell dimensions are derived from the actual
            image sizes rather than expanding to fill ``max_resolution``.
            This minimizes wasted letterboxing / pillarboxing space.
            ``max_resolution`` still acts as an upper bound.  Defaults to
            ``False``.
        background_color: ``(R, G, B)`` tuple for the background / padding
            area.  Defaults to white ``(255, 255, 255)``.
        label_font_scale: OpenCV ``FONT_HERSHEY_SIMPLEX`` scale factor for
            per-image labels.  Defaults to ``0.7``.
        label_color: ``(R, G, B)`` tuple for the label text.  Defaults to
            white ``(255, 255, 255)``.
        label_thickness: Thickness in pixels for the label text glyphs.
            Defaults to ``2``.

    Example::

        config = GridConfig(padding=30, background_color=(0, 0, 0))
        gen = GridGenerator(config)

        # Tweak a single parameter after construction
        gen.config.label_font_scale = 1.0
    """

    padding: int = 20
    max_resolution: int = 1920
    tight: bool = False
    outer_padding: bool = True
    background_color: Tuple[int, int, int] = (255, 255, 255)
    label_font_scale: float = 0.7
    label_color: Tuple[int, int, int] = (255, 255, 255)
    label_thickness: int = 2


# ---------------------------------------------------------------------------
# Grid generator
# ---------------------------------------------------------------------------


class GridGenerator:
    """Compose multiple images into a uniform grid with labels and padding.

    Images are added one at a time via :meth:`add_image` and then arranged
    into a ``rows × cols`` grid in **row-major order** (left to right, top
    to bottom) when :meth:`generate` is called.

    Every cell in the grid has identical dimensions.  Each source image is
    resized to fit its cell while preserving its original aspect ratio;
    letterboxing / pillarboxing fills any unused space with the configured
    background color.

    Optional text labels are overlaid at the top of each cell with a
    semi-transparent dark strip behind the text for readability.

    All methods return a **new** array and leave the stored images unchanged.

    Example::

        import cv2

        gen = GridGenerator(padding=10)

        # Read images and convert to RGB (GridGenerator expects RGB input)
        img_a = cv2.cvtColor(cv2.imread("a.jpg"), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread("b.jpg"), cv2.COLOR_BGR2RGB)
        img_c = cv2.cvtColor(cv2.imread("c.jpg"), cv2.COLOR_BGR2RGB)

        gen.add_image(img_a, label="Input")
        gen.add_image(img_b, label="Prediction")
        gen.add_image(img_c, label="Ground Truth")
        output = gen.generate(cols=3)          # 1-row × 3-col grid

        # Write output (cv2.imwrite expects BGR, so convert back)
        cv2.imwrite("comparison.jpg", cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        config: Optional[GridConfig] = None,
        **kwargs,
    ) -> None:
        """Initialize the grid generator.

        You can supply a pre-built :class:`GridConfig` **or** pass individual
        config fields as keyword arguments — but not both.

        Args:
            config: A :class:`GridConfig` instance that fully describes the
                rendering style.  When provided, ``kwargs`` must be empty.
            **kwargs: Shorthand for any :class:`GridConfig` field, e.g.
                ``padding=10`` or ``background_color=(0, 0, 0)``.  Ignored
                when ``config`` is supplied.

        Raises:
            ValueError: If both ``config`` and ``kwargs`` are provided.
            TypeError: If any kwarg does not match a :class:`GridConfig`
                field (raised by the dataclass constructor).

        Example::

            # Default settings
            gen = GridGenerator()

            # Via keyword arguments
            gen = GridGenerator(padding=30, max_resolution=1024)

            # Via a config object
            config = GridConfig(padding=10)
            gen = GridGenerator(config)
        """
        if config is not None and kwargs:
            raise ValueError(
                "Provide either config or keyword arguments, not both."
            )
        self.config = config if config is not None else GridConfig(**kwargs)

        self._images: List[np.ndarray] = []
        self._labels: List[Optional[str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_image(
        self,
        image: np.ndarray,
        label: Optional[str] = None,
    ) -> "GridGenerator":
        """Add an image to the next available grid cell.

        Images are placed in the order they are added — the first image
        goes to cell ``(row=0, col=0)``, the second to ``(0, 1)``, and so
        on, filling the grid row by row.

        Args:
            image: A numpy array of shape ``(H, W)`` (grayscale) or
                ``(H, W, C)`` (colour), dtype ``uint8``.  4-channel
                (RGBA) images are converted to RGB by dropping the alpha
                channel.  Single-channel images are broadcast to 3
                channels.
            label: Optional text drawn at the top of the cell when the
                grid is generated.  Pass ``None`` (the default) to omit.

        Returns:
            ``self`` so that calls may be chained::

                gen.add_image(a, "A").add_image(b, "B").generate(rows=1, cols=2)
        """
        self._images.append(image)
        self._labels.append(label)
        return self

    def generate(
        self,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
    ) -> np.ndarray:
        """Assemble and return the grid image.

        Grid dimensions are resolved as follows:

        * Both *rows* and *cols* given → use as-is.
        * Only *rows* given → ``cols = ceil(num_images / rows)``.
        * Only *cols* given → ``rows = ceil(num_images / cols)``.
        * Neither given → automatically choose dimensions that make the
          grid as square as possible.

        If there are fewer images than ``rows × cols``, the extra cells
        are filled with the background colour.

        Args:
            rows: Number of rows.  Auto-calculated when ``None``.
            cols: Number of columns.  Auto-calculated when ``None``.

        Returns:
            A numpy array of shape ``(H, W, 3)`` and dtype ``uint8``
            containing the assembled grid in **RGB** colour order.

        Raises:
            ValueError: If no images have been added.
        """
        if not self._images:
            raise ValueError("No images added.  Call add_image() first.")

        n = len(self._images)
        rows, cols = self._resolve_grid_dims(n, rows, cols)

        cell_w, cell_h = self._compute_cell_size(rows, cols)
        canvas = self._make_canvas(rows, cols, cell_w, cell_h)

        for idx in range(min(n, rows * cols)):
            img = self._images[idx]
            label = self._labels[idx] if idx < len(self._labels) else None
            r, c = divmod(idx, cols)
            cell = self._render_cell(img, label, cell_w, cell_h)
            self._blit_cell(canvas, cell, r, c, cell_w, cell_h)

        return canvas

    def clear(self) -> "GridGenerator":
        """Remove all images so the instance can be reused.

        Returns:
            ``self`` for method chaining.

        Example::

            gen = GridGenerator()
            gen.add_image(img1).add_image(img2)
            grid1 = gen.generate(rows=1, cols=2)

            gen.clear()
            gen.add_image(img3).add_image(img4).add_image(img5)
            grid2 = gen.generate(cols=3)
        """
        self._images.clear()
        self._labels.clear()
        return self

    @property
    def num_images(self) -> int:
        """Number of images currently staged (read-only)."""
        return len(self._images)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_grid_dims(
        n: int,
        rows: Optional[int],
        cols: Optional[int],
    ) -> Tuple[int, int]:
        """Resolve *rows* and *cols* from explicit / auto values.

        Args:
            n: Number of images.
            rows: Explicit row count, or ``None``.
            cols: Explicit column count, or ``None``.

        Returns:
            ``(rows, cols)`` tuple, both ≥ 1.

        Raises:
            ValueError: If the resolved grid cannot hold all *n* images.
        """
        if rows is not None and cols is not None:
            if rows * cols < n:
                raise ValueError(
                    f"{rows}×{cols} grid ({rows * cols} cells) is too small "
                    f"for {n} images."
                )
            return rows, cols

        if rows is not None:
            if rows < 1:
                raise ValueError(f"rows must be ≥ 1; got {rows}")
            return rows, (n + rows - 1) // rows

        if cols is not None:
            if cols < 1:
                raise ValueError(f"cols must be ≥ 1; got {cols}")
            return (n + cols - 1) // cols, cols

        # Auto: make the grid as close to square as practical.
        cols = int(np.ceil(np.sqrt(n)))
        rows = (n + cols - 1) // cols
        return rows, cols

    def _compute_cell_size(
        self, rows: int, cols: int
    ) -> Tuple[int, int]:
        """Return the ``(width, height)`` of every cell in pixels.

        The cell dimensions are chosen so that the whole grid (including
        padding) fits within :attr:`GridConfig.max_resolution` on both
        axes.

        When :attr:`GridConfig.tight` is ``True``, the cell size is derived
        from the actual image dimensions rather than expanding to fill
        ``max_resolution`` — this minimises letterboxing / pillarboxing
        when images share a similar aspect ratio.

        The result is always at least 1 px in each dimension.
        """
        pad = self.config.padding
        max_dim = self.config.max_resolution

        if self.config.tight and self._images:
            return self._compute_tight_cell_size(rows, cols)

        # Padding gaps: N+1 with outer padding, N-1 without
        pad_cols = cols + 1 if self.config.outer_padding else cols - 1
        pad_rows = rows + 1 if self.config.outer_padding else rows - 1

        avail_w = max_dim - max(pad_cols, 0) * pad
        avail_h = max_dim - max(pad_rows, 0) * pad
        cell_w = max(1, avail_w // cols)
        cell_h = max(1, avail_h // rows)
        return cell_w, cell_h

    def _compute_tight_cell_size(
        self, rows: int, cols: int
    ) -> Tuple[int, int]:
        """Derive cell size from the actual images, capped at *max_resolution*.

        The "natural" cell is sized to hold the largest image (by width and
        by height) among the staged images.  If the resulting grid would
        exceed ``max_resolution`` on either axis, both cell dimensions are
        scaled down by the same factor so the grid fits.
        """
        pad = self.config.padding
        max_dim = self.config.max_resolution
        outer = self.config.outer_padding

        # --- natural cell = bounding box of the largest image ---
        max_img_w = max_img_h = 1
        for img in self._images:
            h, w = img.shape[:2]
            if w > max_img_w:
                max_img_w = w
            if h > max_img_h:
                max_img_h = h

        cell_w, cell_h = max_img_w, max_img_h

        # --- scale down uniformly if the grid exceeds max_resolution ---
        pad_cols = cols + 1 if outer else cols - 1
        pad_rows = rows + 1 if outer else rows - 1
        total_w = cols * cell_w + max(pad_cols, 0) * pad
        total_h = rows * cell_h + max(pad_rows, 0) * pad

        if total_w > max_dim or total_h > max_dim:
            needed_cell_w = max_dim - max(pad_cols, 0) * pad
            needed_cell_h = max_dim - max(pad_rows, 0) * pad
            if needed_cell_w < 1 or needed_cell_h < 1:
                return 1, 1
            scale = min(needed_cell_w / (cols * cell_w),
                        needed_cell_h / (rows * cell_h))
            cell_w = max(1, int(cell_w * scale))
            cell_h = max(1, int(cell_h * scale))

        return cell_w, cell_h

    def _make_canvas(
        self, rows: int, cols: int, cell_w: int, cell_h: int
    ) -> np.ndarray:
        """Allocate the full output array, filled with the background colour."""
        pad = self.config.padding
        outer = self.config.outer_padding
        pad_cols = cols + 1 if outer else cols - 1
        pad_rows = rows + 1 if outer else rows - 1
        total_w = cols * cell_w + max(pad_cols, 0) * pad
        total_h = rows * cell_h + max(pad_rows, 0) * pad
        return np.full(
            (total_h, total_w, 3),
            self.config.background_color,
            dtype=np.uint8,
        )

    def _render_cell(
        self,
        image: np.ndarray,
        label: Optional[str],
        cell_w: int,
        cell_h: int,
    ) -> np.ndarray:
        """Prepare a single grid cell.

        Steps:
        1. Normalise the image to 3-channel RGB ``uint8``.
        2. Resize to fit *cell_w* × *cell_h*, preserving aspect ratio.
        3. Center the resized image on a background-colour canvas.
        4. Optionally overlay a label strip at the top.
        """
        image = self._to_rgb(image)
        cell = np.full(
            (cell_h, cell_w, 3),
            self.config.background_color,
            dtype=np.uint8,
        )

        # --- resize to fit, preserving aspect ratio ---
        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return cell  # degenerate input → empty cell

        scale = min(cell_w / w, cell_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 1 or new_h < 1:
            return cell

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # --- center ---
        y_off = (cell_h - new_h) // 2
        x_off = (cell_w - new_w) // 2
        cell[y_off : y_off + new_h, x_off : x_off + new_w] = resized

        # --- label overlay ---
        if label:
            self._draw_label(cell, label)

        return cell

    def _draw_label(self, cell: np.ndarray, text: str) -> None:
        """Draw *text* with a dark semi-transparent strip at the top of *cell*.

        The method mutates *cell* in place.  Because OpenCV drawing functions
        expect colours in **BGR** order while the rest of the pipeline works
        in RGB, we convert the configured ``(R, G, B)`` colours to
        ``(B, G, R)`` here.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.config.label_font_scale
        thickness = self.config.label_thickness
        # Convert (R, G, B) → (B, G, R) for cv2.putText / cv2.rectangle
        rgb = self.config.label_color
        bgr = (rgb[2], rgb[1], rgb[0])

        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        strip_h = th + baseline + 8  # small vertical padding

        # Semi-transparent dark strip (black is the same in RGB and BGR)
        overlay = cell.copy()
        cv2.rectangle(overlay, (0, 0), (cell.shape[1], strip_h), (0, 0, 0), -1)
        cv2.addWeighted(cell, 0.4, overlay, 0.6, 0, dst=cell)

        # Text centred horizontally
        tx = max(0, (cell.shape[1] - tw) // 2)
        ty = th + 4  # baseline offset from top of strip
        cv2.putText(
            cell, text, (tx, ty),
            font, scale, bgr, thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """Normalise *image* to 3-channel RGB ``uint8``.

        - Grayscale ``(H, W)`` → broadcast to 3 channels.
        - RGBA ``(H, W, 4)`` → drop alpha, keep RGB.
        - Already 3-channel → returned as-is (caller is responsible for
          ensuring the colour order is RGB).
        """
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        if image.ndim == 3 and image.shape[2] == 3:
            return image
        raise ValueError(
            f"Unsupported image shape {image.shape}. "
            f"Expected (H, W), (H, W, 3), or (H, W, 4)."
        )

    def _blit_cell(
        self,
        canvas: np.ndarray,
        cell: np.ndarray,
        row: int,
        col: int,
        cell_w: int,
        cell_h: int,
    ) -> None:
        """Copy *cell* into the correct position on *canvas*."""
        pad = self.config.padding
        outer = 1 if self.config.outer_padding else 0
        y = outer * pad + row * (cell_h + pad)
        x = outer * pad + col * (cell_w + pad)
        canvas[y : y + cell_h, x : x + cell_w] = cell
