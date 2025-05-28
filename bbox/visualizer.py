import numpy as np

from PIL import Image, ImageDraw
from typing import List, Tuple, Union


class Visualizer:
    def __init__(self):
        pass

    def normalize_bboxes(
        self,
        bboxes: Union[
            List[Tuple[int, int, int, int]], List[int], Tuple[int, int, int, int]
        ],
    ) -> List[Tuple[int, int, int, int]]:
        """
        Normalize bounding boxes into a list of tuples (x1, y1, x2, y2).

        Args:
            bboxes (Union[List[Tuple[int, int, int, int]], List[int], Tuple[int, int, int, int]]):
                Bounding boxes in various input formats.

        Returns:
            List[Tuple[int, int, int, int]]: Normalized list of bounding boxes.

        Raises:
            ValueError: If the bounding box format is not supported.
        """
        if isinstance(bboxes, tuple):
            # Single bounding box as a tuple
            return [bboxes]

        if isinstance(bboxes, list):
            if all(isinstance(b, (list, tuple)) for b in bboxes):
                # List of bounding boxes as tuples/lists
                return [tuple(b) for b in bboxes]
            elif all(isinstance(b, int) for b in bboxes) and len(bboxes) == 4:
                # Single bounding box as a flat list
                return [tuple(bboxes)]

        raise ValueError(
            "Invalid bounding box format. Expected a tuple (x1, y1, x2, y2), "
            "a list of such tuples, or a flat list of 4 integers."
        )

    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        bboxes: Union[
            List[Tuple[int, int, int, int]], List[int], Tuple[int, int, int, int]
        ],
        colors: Union[Tuple[int, int, int], List[Tuple[int, int, int]]] = (255, 0, 0),
        labels: List[str] = None,
        width: int = 3,
    ) -> np.ndarray:
        """
        Draw bounding boxes on the image.

        Args:
            image (np.ndarray): The input image as a numpy array.
            bboxes (Union[List[Tuple[int, int, int, int]], List[int], Tuple[int, int, int, int]]): Bounding boxes to draw.
                Each bounding box is represented as a tuple of (x1, y1, x2, y2).
            colors (Union[Tuple[int, int, int], List[Tuple[int, int, int]]]): Color for the bounding boxes.
            labels (List[str], optional): List of labels for each bounding box. Defaults to None.
            width (int): Line width of the bounding boxes. Defaults to 3.

        Returns:
            np.ndarray: The image with bounding boxes drawn on it.
        """
        # Normalize the bounding boxes
        bboxes = self.normalize_bboxes(bboxes)

        if len(bboxes) == 0:
            return np.array(image)

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)

        # Check if colors is a single color or a list of colors
        # single color: tuple of integer or list of integers
        if isinstance(colors, tuple) or isinstance(colors, list) and len(colors) == 3:
            colors = [colors] * len(bboxes)

        assert len(bboxes) == len(
            colors
        ), "Number of bounding boxes must match number of colors."
        if labels is not None:
            assert len(bboxes) == len(
                labels
            ), "Number of bounding boxes must match number of labels."
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            color = tuple(colors[i])
            draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

            if labels is not None and i < len(labels):
                label = labels[i]
                draw.text((x1, y1), label, fill=color)

        return np.array(image)
