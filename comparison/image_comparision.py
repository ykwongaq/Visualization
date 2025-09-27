import os
import cv2
import numpy as np
import argparse

from typing import List


def is_image_file(filename: str) -> bool:
    try:
        img = cv2.imread(filename)
        return img is not None
    except Exception:
        return False


def gen_image(
    images: List[np.ndarray],
    labels: List[str],
    label_fontsize: int,
    label_height: int = 40,
    padding: int = 10,
    rows=1,
    cols=None,
) -> np.ndarray:
    """
    Generate a single image by combining multiple images with labels.

    Args:
        image (List[np.ndarray]): List of images to be combined.
        labels (List[str]): List of labels corresponding to each image.
        label_fontsize (int): Font size for the labels.
        label_height (int, optional): Height of the label area. Defaults to 40.
        padding (int, optional): Padding between images. Defaults to 10.
        rows (int, optional): Number of rows in the final image. Defaults to 1.
        cols (int, optional): Number of columns in the final image. If None, calculated based on number of images and rows. Defaults to None.
    """
    num_images = len(images)
    if cols is None:
        cols = int(np.ceil(num_images / rows))
    total_slots = rows * cols

    # Reference size from first image
    ref_h, ref_w = images[0].shape[:2]

    def resize(img):
        return cv2.resize(img, (ref_w, ref_h))

    # Prepare fonts
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = label_fontsize / 30.0  # heuristic scaling
    font_thickness = 2
    text_color = (255, 255, 255)  # white

    # Final canvas size
    canvas_h = rows * (ref_h + label_height) + (rows - 1) * padding
    canvas_w = cols * ref_w + (cols - 1) * padding
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for idx in range(total_slots):
        row = idx // cols
        col = idx % cols
        x0 = col * (ref_w + padding)
        y0 = row * (ref_h + label_height + padding)

        block = np.zeros((ref_h + label_height, ref_w, 3), dtype=np.uint8)

        if idx < num_images:
            img = resize(images[idx])
            block[label_height:, :, :] = img

            label = labels[idx] if labels else f"Img {idx+1}"
            (tw, th), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )
            text_x = (ref_w - tw) // 2
            text_y = (label_height + th) // 2
            cv2.putText(
                block,
                label,
                (text_x, text_y),
                font,
                font_scale,
                text_color,
                font_thickness,
                lineType=cv2.LINE_AA,
            )

        canvas[y0 : y0 + ref_h + label_height, x0 : x0 + ref_w] = block

    return canvas


def main(args):
    image_dirs = args.image_dirs
    labels = args.labels
    output_dir = args.output_dir
    label_fontsize = args.label_fontsize
    label_height = args.label_height
    padding = args.padding
    rows = args.rows
    cols = args.cols

    os.makedirs(output_dir, exist_ok=True)

    image_files_list = []
    for image_dir in image_dirs:
        image_files = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                image_file = os.path.join(root, file)
                if is_image_file(image_file):
                    image_files.append(image_file)

        image_files.sort()
        image_files_list.append(image_files)

    num_images = len(image_files_list[0])
    for idx, image_files in enumerate(image_files_list):
        assert (
            len(image_files) == num_images
        ), f"All directories must have the same number of images. But got {len(image_files)} for dir {image_dirs[idx]} and {num_images} for dir {image_dirs[0]}"

    for i in range(num_images):
        images = []
        for image_files in image_files_list:
            img = cv2.imread(image_files[i])
            images.append(img)

        combined_image = gen_image(
            images,
            labels,
            label_fontsize,
            label_height,
            padding,
            rows,
            cols,
        )

        output_path = os.path.join(output_dir, f"{i:04d}.png")
        cv2.imwrite(output_path, combined_image)
        print(f"Saved combined image to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dirs",
        type=str,
        nargs="+",
        required=True,
        help="List of directories containing images to compare.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        required=True,
        help="List of labels for each image directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the combined images.",
    )
    parser.add_argument(
        "--label_fontsize",
        type=int,
        default=20,
        help="Font size for the labels.",
    )
    parser.add_argument(
        "--label_height",
        type=int,
        default=40,
        help="Height of the label area.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding between images.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of rows in the final image.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of columns in the final image. If None, calculated based on number of images and rows.",
    )
    args = parser.parse_args()
    main(args)
