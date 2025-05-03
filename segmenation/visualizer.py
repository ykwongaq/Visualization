import numpy as np
import cv2

from typing import Dict, Tuple

COLOR_MAPS = {}

COLOR_MAPS = {
    0: [128, 0, 128],  # Purple
    1: [255, 165, 0],  # Orange
    2: [255, 192, 203],  # Pink
    3: [165, 42, 42],  # Brown
    4: [50, 205, 50],  # Lime Green
    5: [0, 128, 128],  # Teal
    6: [128, 128, 0],  # Olive
    7: [128, 0, 0],  # Maroon
    8: [0, 0, 128],  # Navy
    9: [255, 215, 0],  # Gold
    10: [192, 192, 192],  # Silver
    11: [64, 224, 208],  # Turquoise
    12: [75, 0, 130],  # Indigo
    13: [230, 230, 250],  # Lavender
    14: [245, 245, 220],  # Beige
    15: [255, 127, 80],  # Coral
    16: [127, 255, 212],  # Aquamarine
    17: [220, 20, 60],  # Crimson
    18: [240, 230, 140],  # Khaki
    19: [250, 128, 114],  # Salmon
    20: [221, 160, 221],  # Plum
    21: [127, 255, 0],  # Chartreuse
    22: [204, 204, 255],  # Periwinkle
    23: [152, 255, 152],  # Mint Green
    24: [210, 105, 30],  # Chocolate
    25: [139, 69, 19],  # Saddle Brown
    26: [233, 150, 122],  # Dark Salmon
    27: [72, 61, 139],  # Dark Slate Blue
    28: [143, 188, 143],  # Dark Sea Green
    29: [255, 140, 0],  # Dark Orange
}

COLOR_MAPS_WIHT_BG = {
    0: [0, 0, 0],  # Black (Background)
    1: [128, 0, 128],  # Purple
    2: [255, 165, 0],  # Orange
    3: [255, 192, 203],  # Pink
    4: [165, 42, 42],  # Brown
    5: [50, 205, 50],  # Lime Green
    6: [0, 128, 128],  # Teal
    7: [128, 128, 0],  # Olive
    8: [128, 0, 0],  # Maroon
    9: [0, 0, 128],  # Navy
    10: [255, 215, 0],  # Gold
    11: [192, 192, 192],  # Silver
    12: [64, 224, 208],  # Turquoise
    13: [75, 0, 130],  # Indigo
    14: [230, 230, 250],  # Lavender
    15: [245, 245, 220],  # Beige
    16: [255, 127, 80],  # Coral
    17: [127, 255, 212],  # Aquamarine
    18: [220, 20, 60],  # Crimson
    19: [240, 230, 140],  # Khaki
    20: [250, 128, 114],  # Salmon
    21: [221, 160, 221],  # Plum
    22: [127, 255, 0],  # Chartreuse
    13: [204, 204, 255],  # Periwinkle
    24: [152, 255, 152],  # Mint Green
    25: [210, 105, 30],  # Chocolate
    26: [139, 69, 19],  # Saddle Brown
    27: [233, 150, 122],  # Dark Salmon
    28: [72, 61, 139],  # Dark Slate Blue
    29: [143, 188, 143],  # Dark Sea Green
    30: [255, 140, 0],  # Dark Orange
}


class Visualizer:
    """
    Visualizer is used to visualize the segmentation masks on the images.
    """

    # Default boundary width is the ratio of the image shortest side
    DEFAULT_BOUNDARY_RATIO = 0.005

    DEFAULT_MASK_ALPHA = 0.3
    DEFAULT_BUNDARY_ALPHA = 0.5


    def __init__(self, mask_alpha: float = DEFAULT_MASK_ALPHA, color_map: Dict = COLOR_MAPS, boundary_alpha: float = DEFAULT_BUNDARY_ALPHA, boundary_ratio: float = DEFAULT_BOUNDARY_RATIO):
        self.mask_alpha = mask_alpha
        self.color_map = color_map
        self.boundary_alpha = boundary_alpha
        self.boundary_width_ratio = boundary_ratio

    def set_color_map(self, color_map: Dict ):
        """
        Set the color map for visualization.

        Args:
            color_map (Dict): A dictionary mapping class indices to colors.
        """
        self.color_map = color_map

    def set_boundary_width(self, boundary_width: int):
        """
        Set the boundary width for visualization.

        Args:
            boundary_width (int): The width of the boundary in pixels.
        """
        self.boundary_width = boundary_width

    def set_mask_alpha(self, mask_alpha: float):
        """
        Set the alpha value for the mask overlay.

        Args:
            mask_alpha (float): The alpha value for the mask overlay.
        """
        self.mask_alpha = mask_alpha

    def set_boundary_alpha(self, boundary_alpha: float):
        """
        Set the alpha value for the boundary overlay.

        Args:
            boundary_alpha (float): The alpha value for the boundary overlay.
        """
        self.boundary_alpha = boundary_alpha

    def set_boundary_width_ratio(self, boundary_width_ratio: float):
        """
        Set the boundary width ratio for visualization.

        Args:
            boundary_width_ratio (float): The ratio of the image shortest side to determine the boundary width.
        """
        self.boundary_width_ratio = boundary_width_ratio

    def visualize_mask(self, image: np.ndarray, mask: np.ndarray, category_id: int = 0, color: Tuple[int, int, int] = None, add_boundary: bool = True) -> np.ndarray:
        """
        Visualize the segmentation mask on the image.

        Args:
            image (np.ndarray): The input image.
            mask (np.ndarray): Binary mask.
            category_id (int): The category ID for the mask.
            color (Tuple[int, int, int]): The color for the mask overlay. If not set, then we use the default color map.
            add_boundary (bool): Whether to add a boundary around the mask.
        Returns:
            np.ndarray: The image with the mask overlayed.
        """
        assert image.ndim == 3, f"Image must be a 3D array (H, W, C), but got {image.ndim}D"
        assert mask.ndim == 2, f"Mask must be a 2D array (H, W), but got {mask.ndim}D"
        assert image.shape[:2] == mask.shape, f"Image and mask must have the same spatial dimensions, but got {image.shape[:2]} for image and {mask.shape} for mask"

        # Ensure the mask is binary
        mask = (mask > 0).astype(np.uint8)

        # Create a color version of the mask
        if color is None:
            color = self.color_map[category_id % len(self.color_map)]

        color_mask = np.zeros_like(image)
        color_mask[mask == 1] = color

        color_mask = cv2.addWeighted(image, 1 - self.mask_alpha, color_mask, self.mask_alpha, 0)
        output_image = np.where(mask[:, :, None] == 1, color_mask, image)

        if add_boundary:
            output_image = self.visualize_boundary(output_image, mask, category_id=category_id, color=color)

        return output_image
    
    def visualize_boundary(self, image: np.ndarray, mask: np.ndarray, category_id: int = 0, color:Tuple[int, int, int] = None, boundary_width: int = None) -> np.ndarray:
        """
        Add a boundary around the mask.

        Args:
            image (np.ndarray): The input image.
            mask (np.ndarray): Binary mask.
            category_id (int): The category ID for the mask.
        Returns:
            np.ndarray: The image with the boundary overlayed.
        """
        assert image.ndim == 3, f"Image must be a 3D array (H, W, C), but got {image.ndim}D"
        assert mask.ndim == 2, f"Mask must be a 2D array (H, W), but got {mask.ndim}D"
        assert image.shape[:2] == mask.shape, f"Image and mask must have the same spatial dimensions, but got {image.shape[:2]} for image and {mask.shape} for mask"

        # Ensure the mask is binary
        mask = (mask > 0).astype(np.uint8)
        image_h, image_w = image.shape[:2]

        if color is None:
            color = self.color_map[category_id % len(self.color_map)]
        
        if boundary_width is None:
            # Calculate the boundary width based on the shortest side of the image
            boundary_width = int(min(image_h, image_w) * self.boundary_width_ratio)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (boundary_width, boundary_width))
        eroded_mask = cv2.erode(mask, kernel, iterations=1)
        boundary = mask - eroded_mask

        boundary_mask = np.zeros_like(image, dtype=np.uint8)
        boundary_mask[boundary == 1] = color

        boundary_mask = cv2.addWeighted(image, 1 - self.boundary_alpha, boundary_mask, self.boundary_alpha, 0)
        output_image = np.where(boundary[:, :, None] == 1, boundary_mask, image)
        
        return output_image
    
