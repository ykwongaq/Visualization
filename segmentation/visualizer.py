import numpy as np
import cv2

from typing import Dict, Tuple, Set

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

class Visualizer:
    """
    Visualizer is used to visualize the segmentation masks on the images.
    """

    # Default boundary width is the ratio of the image shortest side
    DEFAULT_BOUNDARY_RATIO = 0.007

    DEFAULT_MASK_ALPHA = 0.4
    DEFAULT_BUNDARY_ALPHA = 0.7


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

    def visualize_binary_mask(self, image: np.ndarray, mask: np.ndarray, category_id: int = 0, color: Tuple[int, int, int] = None, add_boundary: bool = True) -> np.ndarray:
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
    
    def visualize_sementic_mask(self, image: np.ndarray, mask: np.ndarray, ignore_idx: Set = set([0])) -> np.ndarray:
        """
        Visualize the segmentation mask on the image.
        Args:
            image (np.ndarray): The input image.
            mask (np.ndarray): Semantic mask with class indices.
            ignore_idx (Set): The class indices to ignore.

        Returns:
            np.ndarray: The image with the mask overlayed.
        """

        assert image.ndim == 3, f"Image must be a 3D array (H, W, C), but got {image.ndim}D"
        assert mask.ndim == 2, f"Mask must be a 2D array (H, W), but got {mask.ndim}D"
        assert image.shape[:2] == mask.shape, f"Image and mask must have the same spatial dimensions, but got {image.shape[:2]} for image and {mask.shape} for mask"
        
        output_image = image.copy()
        unique_classes = np.unique(mask)
        for class_id in unique_classes:
            if class_id in ignore_idx:
                continue
            category_mask = (mask == class_id).astype(np.uint8)
            output_image = self.visualize_binary_mask(output_image, category_mask, category_id=class_id)
        return output_image