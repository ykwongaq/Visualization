import numpy as np
import cv2

from typing import List, Dict, Tuple

COLOR_MAPS = {
    0: [135, 206, 250],  # Light Sky Blue
    1: [255, 182, 193],  # Light Pink
    2: [152, 251, 152],  # Pale Green
    3: [255, 160, 122],  # Light Salmon
    4: [238, 130, 238],  # Violet
    5: [240, 230, 140],  # Khaki
    6: [250, 128, 114],  # Salmon
    7: [221, 160, 221],  # Plum
    8: [175, 238, 238],  # Pale Turquoise
    9: [255, 228, 181],  # Moccasin
    10: [144, 238, 144],  # Light Green
    11: [216, 191, 216],  # Thistle
    12: [255, 218, 185],  # Peach Puff
    13: [32, 178, 170],  # Light Sea Green
    14: [100, 149, 237],  # Cornflower Blue
    15: [244, 164, 96],  # Sandy Brown
    16: [72, 209, 204],  # Medium Turquoise
    17: [219, 112, 147],  # Pale Violet Red
    18: [255, 222, 173],  # Navajo White
    19: [255, 105, 180],  # Hot Pink
    20: [255, 127, 80],  # Coral
    21: [173, 216, 230],  # Light Blue
    22: [127, 255, 212],  # Aquamarine
    23: [240, 128, 128],  # Light Coral
    24: [250, 250, 210],  # Light Goldenrod Yellow
    25: [205, 92, 92],  # Indian Red
    26: [176, 224, 230],  # Powder Blue
    27: [210, 180, 140],  # Tan
    28: [255, 239, 213],  # Papaya Whip
    29: [222, 184, 135],  # Burlywood
}


class Visualizer:
    def __init__(self, color_map: Dict = COLOR_MAPS):
        self.color_map = color_map

    def draw_points(
        self,
        image: np.ndarray,
        points: List[Tuple[int]],
        labels: List[int] = None,
        radius_ratio: float = 0.01,
        boundary_width_ratio: float = 0.001,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Visualize points on the image.

        Args:
            image (np.ndarray): The input image in HWC format.
            points (List[Tuple[int]]): List of (x, y) coordinates of points to draw.
            labels (List[int], optional): List of labels corresponding to each point.
                Used to determine the color of the point. Defaults to None.
            radius_ratio (float, optional): Radius of the point as a ratio of the image
                shortest side. Defaults to 0.01.
            boundary_width_ratio (float, optional): Boundary width of the point as a ratio
                of the image shortest side. Defaults to 0.007.
            alpha (float, optional): Alpha blending value for the point overlay. Defaults to 0.5.
        Returns:
            np.ndarray: Image with points drawn.
        """
        output = image.copy()
        h, w = output.shape[:2]
        min_side = min(w, h)
        radius = max(1, int(min_side * radius_ratio))
        boundary_width = max(1, int(min_side * boundary_width_ratio))

        # Create transparent layer for alpha blending
        overlay = np.zeros_like(output, dtype=np.uint8)

        for i, (x, y) in enumerate(points):
            # Get color
            if labels is not None and i < len(labels):
                color = self.color_map.get(
                    int(labels[i]) % len(self.color_map), COLOR_MAPS[0]
                )
            else:
                color = COLOR_MAPS[0]

            color_bgr = tuple(int(c) for c in color)

            # Draw filled circle on overlay (semi-transparent part)
            cv2.circle(overlay, (int(x), int(y)), radius, color_bgr, -1)

            # Draw boundary directly on output (non-transparent)
            cv2.circle(output, (int(x), int(y)), radius, color_bgr, boundary_width)

        # Blend overlay only where circles were drawn
        mask = (overlay > 0).any(axis=-1)
        blended = output.copy()
        blended[mask] = (
            (1 - alpha) * output[mask].astype(np.float32)
            + alpha * overlay[mask].astype(np.float32)
        ).astype(np.uint8)

        return blended
