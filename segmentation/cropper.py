"""Segmentation mask cropping utilities.

This module provides :func:`crop_image` for extracting a masked region from an
RGB image and returning it as an RGBA image with a transparent background.

Typical usage::

    from segmentation.cropper import crop_image

    rgba = crop_image(image, mask)
"""

import numpy as np


def crop_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Crop the masked region from an RGB image and return it as RGBA.

    The bounding box of the non-zero mask pixels is used to crop both the
    image and the mask. The mask is then used as the alpha channel so pixels
    outside the mask are fully transparent.

    Args:
        image: RGB image of shape ``(H, W, 3)`` and dtype ``uint8``.
        mask: Binary mask of shape ``(H, W)`` or ``(H, W, 1)``. Non-zero
            pixels are treated as foreground.

    Returns:
        RGBA image of shape ``(H', W', 4)`` tightly cropped to the mask
        bounding box, or ``None`` if the mask is empty.

    Raises:
        ValueError: If ``image`` is not a 3-channel array of shape ``(H, W, 3)``.

    Example::

        import cv2
        from segmentation.cropper import crop_image

        image = cv2.cvtColor(cv2.imread("photo.jpg"), cv2.COLOR_BGR2RGB)
        mask  = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)

        rgba = crop_image(image, mask)
        if rgba is not None:
            cv2.imwrite("cropped.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    """
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = (mask > 0).astype(np.uint8)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB (H, W, 3).")

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    cropped_img = image[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

    return np.dstack([
        cropped_img,
        (cropped_mask * 255).astype(np.uint8),
    ])
