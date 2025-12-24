import argparse
import json
import os
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm


def load_config(config_file: str) -> Dict:
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file {config_file} does not exist.")

    with open(config_file, "r") as f:
        config = json.load(f)

    return config


def gen_image_grid(
    images: List[np.ndarray],
    labels: List[str],
    label_fontsize: int,
    label_height: int,
    label_fontthickness: int,
    padding: int,
    layout: List[List[Dict[str, Any]]],
) -> np.ndarray:
    """
    Create an image grid with a black background.
    Labels are white, bold, centered, placed in a reserved area above each image.
    """

    # --- Determine layout grid size ---
    num_rows = len(layout)
    num_cols = 0
    for row in layout:
        col_sum = sum(cell.get("colspan", 1) for cell in row)
        num_cols = max(num_cols, col_sum)

    # --- Determine base cell size from the first image ---
    base_img = images[0]
    base_h, base_w = base_img.shape[:2]
    cell_width = base_w
    cell_height = base_h

    # --- Compute total grid dimensions ---
    total_width = padding * (num_cols + 1) + cell_width * num_cols
    total_height = padding * (num_rows + 1)

    for row in layout:
        row_max_height = 0
        for cell in row:
            img_idx = cell["img"]
            img = images[img_idx]
            colspan = cell.get("colspan", 1)
            rowspan = cell.get("rowspan", 1)
            target_width = cell_width * colspan + padding * (colspan - 1)
            aspect = img.shape[1] / img.shape[0]
            target_height = target_width / aspect
            row_max_height = max(row_max_height, target_height * rowspan)
        total_height += int(row_max_height) + label_height + padding

    # --- Initialize black background ---
    grid_img = np.zeros((int(total_height), int(total_width), 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = label_fontsize / 30
    font_thickness = label_fontthickness  # bold

    # --- Draw images row-by-row ---
    y_offset = padding
    for row in layout:
        x_offset = padding
        row_max_height = 0
        for cell in row:
            img_idx = cell["img"]
            img = images[img_idx]
            colspan = cell.get("colspan", 1)
            rowspan = cell.get("rowspan", 1)

            target_width = cell_width * colspan + padding * (colspan - 1)
            aspect = img.shape[1] / img.shape[0]
            target_height = target_width / aspect

            # Reserve a space for the text area above the image
            img_y_start = y_offset + label_height
            img_y_end = img_y_start + int(target_height)

            # Resize image
            resized = cv2.resize(
                img,
                (int(target_width), int(target_height)),
                interpolation=cv2.INTER_AREA,
            )

            # Paste image below label area
            grid_img[
                int(img_y_start) : int(img_y_end),
                int(x_offset) : int(x_offset + target_width),
            ] = resized

            # --- Draw label above image ---
            label = labels[img_idx] if img_idx < len(labels) else ""
            if label:
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_x = int(x_offset + (target_width - text_size[0]) / 2)
                text_y = int(
                    y_offset + label_height * 0.8
                )  # keep nicely within text area height

                # (Optional) A shadow for better visibility
                cv2.putText(
                    grid_img,
                    label,
                    (text_x + 1, text_y + 1),
                    font,
                    font_scale,
                    (0, 0, 0),
                    font_thickness + 2,
                    cv2.LINE_AA,
                )

                # White bold text
                cv2.putText(
                    grid_img,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    cv2.LINE_AA,
                )

            x_offset += target_width + padding
            row_max_height = max(row_max_height, target_height * rowspan)

        y_offset += row_max_height + label_height + padding

    return grid_img


def gen_image_grid_helper(
    image_files: List[str],
    labels: List[str],
    output_path: str,
    label_fontsize: int,
    label_height: int,
    label_fontthickness: int,
    padding: int,
    layout: List[List[Dict[str, Any]]],
):
    images = []
    for image_file in image_files:
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)

    image_grid = gen_image_grid(
        images,
        labels,
        label_fontsize,
        label_height,
        label_fontthickness,
        padding,
        layout,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR))


def main(args):
    config_file = args.config_file
    config = load_config(config_file)

    image_dirs: List[str] = config["image_dirs"]
    output_dir = config["output_dir"]
    labels: List[str] = config["labels"]
    layout = config["layout"]
    label_fontsize: int = config["label_fontsize"]
    label_height: int = config["label_height"]
    padding: int = config["padding"]
    label_fontthickness: int = config.get("label_fontthickness", 2)

    os.makedirs(output_dir, exist_ok=True)

    image_files_list = []
    for image_dir in image_dirs:
        image_files = sorted(
            [
                os.path.join(root, file)
                for root, _, files in os.walk(image_dir)
                for file in files
                if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
            ]
        )
        image_files_list.append(image_files)

    min_num_images = min(len(files) for files in image_files_list)
    for idx, files in enumerate(image_files_list):
        if len(files) > min_num_images:
            print(
                f"Warning: Directory {image_dirs[idx]} has more images ({len(files)}) than the minimum ({min_num_images}). Truncating."
            )
            image_files_list[idx] = files[:min_num_images]

    num_images = len(image_files_list[0])
    for idx, files in enumerate(image_files_list):
        assert (
            len(files) == num_images
        ), f"All directories must have the same number of images. But got {len(files)} for dir {image_dirs[idx]} and {num_images} for dir {image_dirs[0]}"

    with Pool(processes=cpu_count()) as pool:

        pbar = tqdm(total=num_images, desc="Generating image grids")

        def update(_):
            pbar.update()

        for i in range(num_images):
            images_files = [files[i] for files in image_files_list]

            output_path = os.path.join(output_dir, f"{i:08d}.jpg")
            pool.apply_async(
                gen_image_grid_helper,
                args=(
                    images_files,
                    labels,
                    output_path,
                    label_fontsize,
                    label_height,
                    label_fontthickness,
                    padding,
                    layout,
                ),
                callback=update,
            )

        pool.close()
        pool.join()
        pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="./config.json",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()
    main(args)
