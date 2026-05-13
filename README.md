# Visualization

A collection of Python utilities for computer vision visualization tasks — bounding boxes, segmentation masks, depth maps, point overlays, image grids, and video/frame conversion.

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/ykwongaq/Visualization.git
```

To upgrade to the latest version:

```bash
pip install --upgrade git+https://github.com/ykwongaq/Visualization.git
```

To install a specific release tag:

```bash
pip install git+https://github.com/ykwongaq/Visualization.git@v0.1.0
```

**Requirements:** Python ≥ 3.10

## Modules

### `bbox` — Bounding Boxes

Draw axis-aligned bounding boxes on images with optional text labels.

```python
from bbox import Visualizer, VisualizerConfig

viz = Visualizer(line_width=2, default_color=(0, 255, 0))
output = viz.draw_bounding_boxes(image, bboxes, labels=["cat", "dog"])
```

### `segmentation` — Segmentation Masks

Overlay binary or semantic segmentation masks on images, or crop masked regions.

```python
from segmentation import Visualizer, VisualizerConfig, crop_image

viz = Visualizer()
output = viz.visualize(image, mask)

cropped = crop_image(image, mask)  # returns RGBA image of the masked region
```

### `depth` — Depth Maps

Convert floating-point depth arrays into false-color RGB images using matplotlib colormaps.

```python
from depth import Visualizer, VisualizerConfig

viz = Visualizer(color_map="plasma", reverse=True)
rgb = viz.visualize_depth(depth_array)
```

### `point` — Point Overlays

Draw labeled points as semi-transparent filled circles with a solid boundary stroke.

```python
from point import Visualizer, VisualizerConfig

viz = Visualizer(radius_ratio=0.015, alpha=0.6)
output = viz.draw_points(image, points=[(120, 80), (300, 200)], labels=[0, 2])
```

### `image_grids` — Image Grids

Composite multiple images into a single grid with optional text labels and colspan/rowspan layout support.

```python
from image_grids import gen_grid, GridConfig

config = GridConfig(padding=20, scale=0.5)
layout = [[{"img": 0}], [{"img": 1}]]

gen_grid(
    image_dirs=["path/to/setA", "path/to/setB"],
    output_dir="grids/",
    labels=["Set A", "Set B"],
    layout=layout,
    config=config,
)
```

Or drive it from a JSON config file:

```python
from image_grids import gen_grid_from_config

gen_grid_from_config("config.json")
```

### `video` — Video / Frame Conversion

Extract frames from a video or assemble frames back into a video using ffmpeg.

```python
from video import video_to_frame, frame_to_video, get_fps

video_to_frame("input.mp4", "frames/", fps=24)
frame_to_video("frames/", "output.mp4", fps=24)

fps = get_fps("input.mp4")
```

> **Note:** `ffmpeg` must be installed and available on your `PATH`.
