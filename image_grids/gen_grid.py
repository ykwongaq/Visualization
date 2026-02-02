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
    Create an image grid that supports colspan/rowspan like HTML tables.
    Each image fills all its assigned space including rowspan regions,
    and label height is correctly accounted for in vertical layout.
    """

    num_rows = len(layout)
    num_cols = max(sum(cell.get("colspan", 1) for cell in row) for row in layout)

    base_img = images[0]
    base_h, base_w = base_img.shape[:2]
    cell_w, cell_h = base_w, base_h

    # --- Step 1: Track occupied cells ---
    grid_occupancy = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    for r, row in enumerate(layout):
        c = 0
        for cell in row:
            while c < num_cols and grid_occupancy[r][c] is not None:
                c += 1
            if c >= num_cols:
                raise ValueError(f"Layout overflow in row {r}")
            rowspan = cell.get("rowspan", 1)
            colspan = cell.get("colspan", 1)
            for rr in range(r, r + rowspan):
                for cc in range(c, c + colspan):
                    grid_occupancy[rr][cc] = cell
            c += colspan

    # --- Step 2: Compute pixel offsets including label heights ---
    row_y = [0] * num_rows
    y_offset = padding
    for r in range(num_rows):
        row_y[r] = y_offset
        # each row has image height + label area + padding
        y_offset += cell_h + label_height + padding

    col_x = [padding + c * (cell_w + padding) for c in range(num_cols)]

    total_w = padding * (num_cols + 1) + num_cols * cell_w
    total_h = y_offset  # already includes bottom padding

    grid_img = np.zeros((int(total_h), int(total_w), 3), dtype=np.uint8)

    # --- Font setup ---
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = label_fontsize / 30
    font_thickness = label_fontthickness

    drawn = set()
    for r in range(num_rows):
        for c in range(num_cols):
            cell = grid_occupancy[r][c]
            if cell is None or id(cell) in drawn:
                continue
            drawn.add(id(cell))

            img_idx = cell["img"]
            img = images[img_idx]
            label = labels[img_idx] if img_idx < len(labels) else ""

            colspan = cell.get("colspan", 1)
            rowspan = cell.get("rowspan", 1)

            # --- Compute total area occupied, including label sections ---
            w_px = cell_w * colspan + padding * (colspan - 1)
            h_px = (cell_h + label_height) * rowspan + padding * (rowspan - 1)

            x = col_x[c]
            y = row_y[r]

            # The top label area belongs only to the first row inside the rowspan block
            img_y_start = int(y + label_height)
            img_h = h_px - label_height  # exclude label height only once at the top

            # Resize to fill entire assigned region
            resized = cv2.resize(img, (int(w_px), int(img_h)), interpolation=cv2.INTER_AREA)

            # Paste it in
            grid_img[
                img_y_start : int(img_y_start + img_h),
                int(x) : int(x + w_px)
            ] = resized

            # --- Label above the image (top row only) ---
            if label:
                text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                text_x = int(x + (w_px - text_size[0]) / 2)
                text_y = int(y + label_height * 0.8)
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
    scale: float = 1.0,
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

    if scale != 1.0:
        new_w = int(image_grid.shape[1] * scale)
        new_h = int(image_grid.shape[0] * scale)
        image_grid = cv2.resize(
            image_grid, (new_w, new_h), interpolation=cv2.INTER_AREA
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(image_grid, cv2.COLOR_RGB2BGR))


def gen_grid(config: Dict):
    image_dirs: List[str] = config["image_dirs"]
    output_dir = config["output_dir"]
    labels: List[str] = config["labels"]
    layout = config["layout"]
    label_fontsize: int = config.get("label_fontsize", 80)
    label_height: int = config.get("label_height", 80)
    padding: int = config.get("padding", 10)
    label_fontthickness: int = config.get("label_fontthickness", 5)
    scale: float = config.get("scale", 1.0)

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
                    scale,
                ),
                callback=update,
            )

        pool.close()
        pool.join()
        pbar.close()


def main(args):
    config_file = args.config_file
    config = load_config(config_file)
    gen_grid(config)


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
