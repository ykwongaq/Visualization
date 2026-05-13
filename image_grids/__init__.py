"""Image grid generation package.

Provides tools for compositing multiple images into a single grid image with
optional text labels, colspan/rowspan layout support, and parallel processing.

Modules:
    gen_grid: Generate image grids from lists of images or directories.

Typical usage — from Python::

    from image_grids.gen_grid import gen_grid, GridConfig

    config = GridConfig(padding=20, scale=0.5)
    layout = [[{"img": 0}], [{"img": 1}]]

    gen_grid(
        image_dirs=["path/to/setA", "path/to/setB"],
        output_dir="grids/",
        labels=["Set A", "Set B"],
        layout=layout,
        config=config,
    )

Typical usage — from a JSON config file::

    from image_grids import gen_grid_from_config

    gen_grid_from_config("config.json")
"""

from image_grids.gen_grid import (
    GridConfig,
    gen_grid,
    gen_grid_from_config,
    gen_image_grid,
    gen_single_grid,
    load_config,
)

__all__ = [
    "GridConfig",
    "gen_grid",
    "gen_grid_from_config",
    "gen_image_grid",
    "gen_single_grid",
    "load_config",
]
