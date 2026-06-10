# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] — 2026-06-10

### Fixed

- **`frames_to_video`**: added `pad=ceil(iw/2)*2:ceil(ih/2)*2` video filter so
  frames with odd width or height are padded to even dimensions before encoding,
  preventing ffmpeg failures when using `yuv420p` pixel format.

## [0.2.1] — 2026-06-10

### Changed

- **`GridConfig`**: flipped default values — `tight` now defaults to
  ``True`` (cell sizes derived from actual image dimensions) and
  `outer_padding` now defaults to ``False`` (padding only between cells,
  not at the outer border).  These defaults produce tighter, more
  predictable grids for the common case.

### Fixed

- Reordered imports to follow the standard-library-first convention.
- Normalised line lengths for better readability.

## [0.2.0] — 2026-06-10

### Added

- **`GridGenerator`**: new `tight` parameter — cell dimensions are derived
  from actual image sizes rather than expanding to fill `max_resolution`,
  minimising letterboxing / pillarboxing waste.
- **`GridGenerator`**: new `outer_padding` parameter — when `False`,
  padding only appears between cells, not at the outer border of the grid.

### Changed

- **`image_grids`**: replaced `gen_grid.py` with `grid_generator.py`, a
  more flexible and better-documented grid compositing module.
- **`GridConfig`**: removed unused `config.json`; configuration is now
  entirely through the `GridConfig` dataclass.

### Removed

- **`image_grids`**: deleted `gen_grid.py` and `config.json`.

## [0.1.1] — 2026-06-02

### Changed

- **`frames_to_video`**: the `-start_number` ffmpeg parameter is now
  auto-detected by scanning the input directory for the smallest frame
  number matching the given pattern, instead of being hardcoded to `0`.
  Frames no longer need to start at `00000000`.

### Fixed

- Renamed `frame_to_video` → `frames_to_video` and `video_to_frame` →
  `video_to_frames` for consistent plural naming across the video module.

## [0.1.0] — 2026-03-01

### Added

- `yk_visualization.bbox` — bounding box drawing with optional labels
- `yk_visualization.segmentation` — segmentation mask overlay and cropping
- `yk_visualization.depth` — depth map false-color visualization
- `yk_visualization.point` — labeled point overlays
- `yk_visualization.image_grids` — multi-image grid compositing
- `yk_visualization.video` — ffmpeg-based video ↔ frames conversion
