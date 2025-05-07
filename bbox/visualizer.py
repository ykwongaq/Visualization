import numpy as np

from PIL import Image, ImageDraw
from typing import List, Tuple, Union


class Visualizer:
    def __init__(self):
        pass

    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]],
        labels: List[str] = None,
        width: int = 3,
    ) -> np.ndarray:
        """
        Draw bounding boxes on the image.

        Args:
            image (np.ndarray): The input image as a numpy array.
            bboxes (List[Tuple[int, int, int, int]]): List of bounding boxes in the format (x1, y1, x2, y2).
            colors (Union[Tuple[int, int, int], List[Tuple[int, int, int]]]): Color for the bounding boxes.
            labels (List[str], optional): List of labels for each bounding box. Defaults to None.

        Returns:
            Image.Image: The image with bounding boxes drawn on it.
        """
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        # Check if colors is a single color or a list of colors
        if isinstance(colors, tuple):
            colors = [colors] * len(bboxes)

        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            color = colors[i]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            if labels is not None:
                label = labels[i]
                draw.text((x1, y1), label, fill=color)

        return np.array(image)
