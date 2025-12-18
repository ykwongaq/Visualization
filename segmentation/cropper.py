import cv2
import numpy as np

def crop_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Crop the segmented region from an RGB image and return an RGBA image
    with a transparent background — using only NumPy + OpenCV.

    Args:
        image (np.ndarray): RGB image (H, W, 3)
        mask (np.ndarray): Binary mask (H, W) or (H, W, 1), values 0 or 1

    Returns:
        np.ndarray: RGBA cropped image (H', W', 4)
    """

    # Ensure mask is 2D binary
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask = (mask > 0).astype(np.uint8)

    # Ensure image has 3 channels (RGB)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be RGB (H, W, 3).")

    # Find bounding box of nonzero pixels in the mask
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    # Crop the image and mask to that bounding box
    cropped_img = image[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]

    # Convert RGB to RGBA and assign alpha channel
    rgba = np.dstack([
        cropped_img,                         # RGB channels
        (cropped_mask * 255).astype(np.uint8)  # Alpha channel (0 or 255)
    ])

    return rgba
