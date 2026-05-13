"""Image grid generation utilities.

This module provides :func:`gen_grid` and :func:`gen_image_grid` for
compositing multiple images into a single grid image. The layout supports
HTML-style ``colspan`` and ``rowspan``, so cells can span multiple rows or
columns. Text labels are drawn above each cell with a drop shadow for
legibility.

Grid generation is parallelized across CPU cores via :mod:`multiprocessing`.

Typical usage — from Python::

    from image_grids.gen_grid import gen_grid, GridConfig

    config = GridConfig(padding=20, scale=0.5)
    layout = [[{"img": 0}], [{"img": 1}]]

    gen_grid(
        image_dirs=["path/to/setA", "path/to/setB"],
        output_dir="path/to/output",
        labels=["Set A", "Set B"],
        layout=layout,
        config=config,
    )

Typical usage — from a JSON config file::

    from image_grids.gen_grid import load_config, gen_grid_from_config

    gen_grid_from_config("config.json")

JSON config schema::

    {
        "image_dirs":         ["dir_A", "dir_B"],   // one dir per grid column
        "output_dir":         "out/",
        "labels":             ["Label A", "Label B"],
        "layout":             [[{"img": 0}], [{"img": 1}]],
        "label_fontsize":     80,     // optional, default 80
        "label_height":       80,     // optional, default 80
        "label_fontthickness": 5,     // optional, default 5
        "padding":            10,     // optional, default 10
        "scale":              1.0     // optional, default 1.0
    }
"""

import argparse
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class GridConfig:
    """Tunable rendering parameters for :func:`gen_grid` and :func:`gen_image_grid`.

    All fields have sensible defaults so you only need to specify the ones you
    want to change.

    Attributes:
        label_fontsize: Controls the scale of the OpenCV font used for cell
            labels. The value is divided by 30 to produce an OpenCV
            ``fontScale``. For example, ``60`` → ``fontScale=2.0``.
            Defaults to ``80``.
        label_height: Vertical space in pixels reserved above each image for
            the text label. Set to ``0`` to disable labels entirely.
            Defaults to ``80``.
        label_fontthickness: Stroke thickness of the label text in pixels.
            A shadow stroke of ``thickness + 2`` is drawn first in black for
            legibility. Defaults to ``5``.
        padding: Gap in pixels between cells and around the grid border.
            Defaults to ``10``.
        scale: Uniform scale factor applied to the final grid image before
            saving. ``1.0`` keeps the original resolution; ``0.5`` halves it.
            Defaults to ``1.0``.

    Example::

        # Compact grid saved at half resolution
        config = GridConfig(padding=5, scale=0.5)

        # No labels
        config = GridConfig(label_height=0)
    """

    label_fontsize: int = 80
    label_height: int = 80
    label_fontthickness: int = 5
    padding: int = 10
    scale: float = 1.0


# ---------------------------------------------------------------------------
# Core rendering
# ---------------------------------------------------------------------------

def gen_image_grid(
    images: List[np.ndarray],
    labels: List[str],
    layout: List[List[Dict[str, Any]]],
    config: Optional[GridConfig] = None,
) -> np.ndarray:
    """Composite a list of images into a single grid image.

    The grid layout is described by a nested list that mirrors an HTML table:
    each inner list is a row, each dict in that list is a cell. Cells may
    span multiple columns or rows via ``colspan`` and ``rowspan`` keys.

    All input images are assumed to have the same spatial dimensions (taken
    from ``images[0]``). A cell that spans multiple rows/columns is resized
    to fill the full combined area.

    Args:
        images: RGB images of shape ``(H, W, 3)`` and dtype ``uint8``. All
            images must share the same ``H`` and ``W``.
        labels: Text label for each image, indexed by the ``"img"`` value in
            ``layout``. A cell with no matching label renders without text.
        layout: Nested list describing the grid. Example::

                [
                    [{"img": 0, "colspan": 2}],
                    [{"img": 1}, {"img": 2}],
                ]

            Each cell dict must have an ``"img"`` key (index into ``images``)
            and may have ``"colspan"`` (default ``1``) and ``"rowspan"``
            (default ``1``).
        config: Rendering parameters. When ``None``, :class:`GridConfig`
            defaults are used.

    Returns:
        Composited grid as an RGB image of dtype ``uint8``.

    Raises:
        ValueError: If ``layout`` references more columns than are implied by
            the maximum row width (overflow detection).

    Example::

        layout = [[{"img": 0}], [{"img": 1}]]
        grid = gen_image_grid(images, labels=["A", "B"], layout=layout)
    """
    if config is None:
        config = GridConfig()

    num_rows = len(layout)
    num_cols = max(sum(cell.get("colspan", 1) for cell in row) for row in layout)

    cell_h, cell_w = images[0].shape[:2]

    # Track which (row, col) positions are occupied and by which cell dict
    grid_occupancy = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for r, row in enumerate(layout):
        c = 0
        for cell in row:
            while c < num_cols and grid_occupancy[r][c] is not None:
                c += 1
            if c >= num_cols:
                raise ValueError(f"Layout overflow in row {r}")
            rowspan = cell.get("rowspan", 1)
            colspan = cell.get("colspan", 1)
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    grid_occupancy[rr][cc] = cell
            c += colspan

    # Pixel offsets for each row/column origin
    row_y = [0] * num_rows
    y_offset = config.padding
    for r in range(num_rows):
        row_y[r] = y_offset
        y_offset += cell_h + config.label_height + config.padding

    col_x = [config.padding + c * (cell_w + config.padding) for c in range(num_cols)]

    total_w = config.padding * (num_cols + 1) + num_cols * cell_w
    total_h = y_offset

    grid_img = np.zeros((int(total_h), int(total_w), 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.label_fontsize / 30

    drawn: set = set()
    for r in range(num_rows):
        for c in range(num_cols):
            cell = grid_occupancy[r][c]
            if cell is None or id(cell) in drawn:
                continue
            drawn.add(id(cell))

            img_idx = cell["img"]
            img = images[img_idx]
            label = labels[img_idx] if img_idx < len(labels) else ""

            colspan = cell.get("colspan", 1)
            rowspan = cell.get("rowspan", 1)

            w_px = cell_w * colspan + config.padding * (colspan - 1)
            h_px = (cell_h + config.label_height) * rowspan + config.padding * (rowspan - 1)

            x = col_x[c]
            y = row_y[r]

            # Label area occupies the top of the first row in a rowspan block
            img_y_start = int(y + config.label_height)
            img_h = h_px - config.label_height

            resized = cv2.resize(
                img, (int(w_px), int(img_h)), interpolation=cv2.INTER_AREA
            )
            grid_img[
                img_y_start : int(img_y_start + img_h),
                int(x) : int(x + w_px),
            ] = resized

            if label and config.label_height > 0:
                text_size, _ = cv2.getTextSize(
                    label, font, font_scale, config.label_fontthickness
                )
                text_x = int(x + (w_px - text_size[0]) / 2)
                text_y = int(y + config.label_height * 0.8)
                # Drop shadow
                cv2.putText(
                    grid_img, label,
                    (text_x + 1, text_y + 1),
                    font, font_scale, (0, 0, 0),
                    config.label_fontthickness + 2, cv2.LINE_AA,
                )
                # Foreground text
                cv2.putText(
                    grid_img, label,
                    (text_x, text_y),
                    font, font_scale, (255, 255, 255),
                    config.label_fontthickness, cv2.LINE_AA,
                )

    return grid_img


def _gen_grid_worker(args: Tuple) -> None:
    """Worker function for parallel grid generation.

    Accepts a single tuple so it can be used with :func:`multiprocessing.Pool.imap_unordered`.
    Loads images from disk, composites the grid, optionally scales it, and
    writes the result to ``output_path``.

    Note:
        This function must remain a module-level definition so it can be
        pickled by :mod:`multiprocessing`.
    """
    image_files, labels, output_path, layout, config = args
    images = [
        cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
        for f in image_files
    ]

    grid = gen_image_grid(images, labels, layout, config)

    if config.scale != 1.0:
        new_w = int(grid.shape[1] * config.scale)
        new_h = int(grid.shape[0] * config.scale)
        grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def gen_single_grid(
    image_paths: List[str],
    output_path: str,
    labels: List[str],
    layout: List[List[Dict[str, Any]]],
    config: Optional[GridConfig] = None,
) -> None:
    """Composite a list of image files into a single grid image and save it.

    This is the simplest entry point: provide a flat list of image paths, a
    layout, and an output path, and the function does the rest.  No
    directories or multiprocessing involved — useful whenever you want one
    grid from a fixed set of images.

    Args:
        image_paths: Paths to the source images, one per image slot.  The
            order must match the ``"img"`` indices used in ``layout``.
        output_path: Destination file for the composited grid (e.g.
            ``"out/grid.jpg"``).  Parent directories are created automatically.
        labels: Display label for each image slot, indexed by the ``"img"``
            value in ``layout``.
        layout: Grid layout specification.  See :func:`gen_image_grid` for
            the full format description.
        config: Rendering parameters.  When ``None``, :class:`GridConfig`
            defaults are used.

    Example::

        from image_grids.gen_grid import gen_single_grid, GridConfig

        gen_single_grid(
            image_paths=["a.jpg", "b.jpg", "c.jpg"],
            output_path="grid.jpg",
            labels=["A", "B", "C"],
            layout=[[{"img": 0}, {"img": 1}], [{"img": 2, "colspan": 2}]],
            config=GridConfig(padding=10, scale=0.5),
        )
    """
    if config is None:
        config = GridConfig()

    images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_paths]
    grid = gen_image_grid(images, labels, layout, config)

    if config.scale != 1.0:
        new_w = int(grid.shape[1] * config.scale)
        new_h = int(grid.shape[0] * config.scale)
        grid = cv2.resize(grid, (new_w, new_h), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def gen_grid(
    image_dirs: List[str],
    output_dir: str,
    labels: List[str],
    layout: List[List[Dict[str, Any]]],
    config: Optional[GridConfig] = None,
) -> None:
    """Generate a grid image for every corresponding frame across multiple directories.

    For each index ``i``, the ``i``-th image from every directory in
    ``image_dirs`` is composited into a grid and written to
    ``<output_dir>/<i:08d>.jpg``. Processing is parallelized across all
    available CPU cores.

    All directories must contain the same number of images after truncation
    to the shortest one (a warning is printed when truncation occurs).

    Args:
        image_dirs: List of directories, one per image slot referenced by
            ``"img"`` indices in ``layout``. Images within each directory are
            discovered recursively and sorted by path.
        output_dir: Directory where output grid images are written. Created
            automatically if it does not exist.
        labels: Display label for each image slot, indexed by the ``"img"``
            value in ``layout``.
        layout: Grid layout specification. See :func:`gen_image_grid` for the
            full format description.
        config: Rendering parameters. When ``None``, :class:`GridConfig`
            defaults are used.

    Raises:
        AssertionError: If the image directories (after truncation) do not
            all contain the same number of images.

    Example::

        config = GridConfig(padding=20, scale=0.5)
        layout = [[{"img": 0}], [{"img": 1}]]

        gen_grid(
            image_dirs=["frames/modelA", "frames/modelB"],
            output_dir="grids/",
            labels=["Model A", "Model B"],
            layout=layout,
            config=config,
        )
    """
    if config is None:
        config = GridConfig()

    os.makedirs(output_dir, exist_ok=True)

    _IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

    image_files_list: List[List[str]] = []
    for image_dir in image_dirs:
        files = sorted(
            os.path.join(root, f)
            for root, _, names in os.walk(image_dir)
            for f in names
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        image_files_list.append(files)

    min_count = min(len(f) for f in image_files_list)
    for idx, files in enumerate(image_files_list):
        if len(files) > min_count:
            print(
                f"Warning: '{image_dirs[idx]}' has {len(files)} images but "
                f"the minimum is {min_count}. Truncating."
            )
            image_files_list[idx] = files[:min_count]

    num_images = min_count
    for idx, files in enumerate(image_files_list):
        assert len(files) == num_images, (
            f"Directory count mismatch: '{image_dirs[idx]}' has {len(files)} "
            f"images, expected {num_images}."
        )

    tasks = [
        (
            [files[i] for files in image_files_list],
            labels,
            os.path.join(output_dir, f"{i:08d}.jpg"),
            layout,
            config,
        )
        for i in range(num_images)
    ]

    with Pool(processes=cpu_count()) as pool:
        for _ in tqdm(
            pool.imap_unordered(_gen_grid_worker, tasks),
            total=num_images,
            desc="Generating image grids",
        ):
            pass


def load_config(config_file: str) -> Tuple[Dict[str, Any], GridConfig]:
    """Load a JSON config file and split it into data fields and a :class:`GridConfig`.

    The JSON file may contain any combination of data fields
    (``image_dirs``, ``output_dir``, ``labels``, ``layout``) and style fields
    (``label_fontsize``, ``label_height``, ``label_fontthickness``,
    ``padding``, ``scale``). Style fields are extracted into a
    :class:`GridConfig`; everything else is returned as a plain dict.

    Args:
        config_file: Path to the JSON configuration file.

    Returns:
        A ``(data, config)`` tuple where ``data`` is a dict with the
        non-style fields and ``config`` is a :class:`GridConfig` populated
        from the style fields (missing keys fall back to dataclass defaults).

    Raises:
        FileNotFoundError: If ``config_file`` does not exist.

    Example::

        data, config = load_config("config.json")
        gen_grid(
            image_dirs=data["image_dirs"],
            output_dir=data["output_dir"],
            labels=data["labels"],
            layout=data["layout"],
            config=config,
        )
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file '{config_file}' does not exist.")

    with open(config_file) as f:
        raw = json.load(f)

    _STYLE_KEYS = {"label_fontsize", "label_height", "label_fontthickness", "padding", "scale"}

    grid_config = GridConfig(
        **{k: raw[k] for k in _STYLE_KEYS if k in raw}
    )
    data = {k: v for k, v in raw.items() if k not in _STYLE_KEYS}

    return data, grid_config


def gen_grid_from_config(config_file: str) -> None:
    """Load a JSON config file and run :func:`gen_grid`.

    Convenience wrapper that calls :func:`load_config` and passes the result
    straight to :func:`gen_grid`. Useful as a one-liner for CLI-style usage
    from Python code.

    Args:
        config_file: Path to the JSON configuration file. See
            :func:`load_config` for the expected schema.

    Example::

        from image_grids.gen_grid import gen_grid_from_config

        gen_grid_from_config("config.json")
    """
    data, config = load_config(config_file)
    gen_grid(
        image_dirs=data["image_dirs"],
        output_dir=data["output_dir"],
        labels=data["labels"],
        layout=data["layout"],
        config=config,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(args):
    gen_grid_from_config(args.config_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate image grids from a JSON config file."
    )
    parser.add_argument(
        "--config_file", type=str, default="./config.json",
        help="Path to the JSON configuration file (default: ./config.json).",
    )
    main(parser.parse_args())
