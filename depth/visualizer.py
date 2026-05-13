import argparse
import os

import cv2
import matplotlib
import numpy as np


class DepthVisualizer:
    def __init__(
        self,
        color_map: str = "viridis",
    ):
        # Load the matplotlib colormap once at init time
        self.color_map_name = color_map
        self.cmap = matplotlib.colormaps[color_map]

    def visualize_depth(self, depth: np.ndarray, reverse: bool = False) -> np.ndarray:
        """
        Input the 2D depth map and then output the RGB image
        """
        if depth.ndim != 2:
            raise ValueError(f"Expected a 2D depth map, got shape {depth.shape}")

        # Build a mask of valid pixels (ignore NaN/Inf and non-positive values)
        valid_mask = np.isfinite(depth) & (depth > 0)

        if not np.any(valid_mask):
            # Nothing valid to visualize — return a black image
            h, w = depth.shape
            return np.zeros((h, w, 3), dtype=np.uint8)

        # Normalize using only valid pixels so outliers/invalid don't squash the range
        d_min = depth[valid_mask].min()
        d_max = depth[valid_mask].max()

        normalized_depth = (depth - d_min) / (d_max - d_min + 1e-8)
        normalized_depth = np.clip(normalized_depth, 0.0, 1.0)

        if reverse:
            normalized_depth = 1 - normalized_depth

        # Apply colormap -> returns RGBA float in [0, 1]
        colored = self.cmap(normalized_depth)

        # Drop alpha channel and convert to uint8 RGB
        rgb = (colored[..., :3] * 255).astype(np.uint8)

        # Zero-out invalid pixels (optional, makes them black)
        rgb[~valid_mask] = 0

        return rgb


def main(args):
    depth_path = args.depth_path
    output_path = args.output_path

    visualizer = DepthVisualizer()
    depth = np.load(depth_path)

    vid_depth = visualizer.visualize_depth(depth)
    vis_depth = cv2.cvtColor(vid_depth, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, vis_depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
