# Changelog

All notable changes to this project will be documented in this file.

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
