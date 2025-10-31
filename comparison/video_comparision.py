import argparse
import os
import cv2
import numpy as np


def images_to_video(
    image_lists,
    output_path,
    labels=None,
    fps=5,
    padding=10,
    label_height=40,
    rows=1,
    cols=None,
):
    """
    Combine N lists of images into a grid video with labels and padding.

    Args:
        image_lists (list of list of np.array):
            A list containing N lists of images (each list is from one model).
            Each inner list must have the same length.
        output_path (str): Path to save the output video (e.g., "comparison.mp4").
        labels (list of str or None): Labels to display on top of each video.
        fps (int): Frames per second for the output video.
        padding (int): Padding (in pixels) between videos.
        label_height (int): Height of the label area above each video.
        rows (int): Number of rows in the grid.
        cols (int or None): Number of columns in the grid. If None, it is inferred.
    """
    num_models = len(image_lists)
    num_frames = len(image_lists[0])

    if labels is None:
        labels = [f"Model {i+1}" for i in range(num_models)]
    if len(labels) != num_models:
        raise ValueError("Number of labels must match number of folders/models")

    # Ensure every model has same number of frames
    for lst in image_lists:
        if len(lst) != num_frames:
            raise ValueError("All image lists must have the same number of frames")

    # Resize all images to same shape (take first image of first list as reference)
    ref_h, ref_w = image_lists[0][0].shape[:2]

    def resize(img):
        return cv2.resize(img, (ref_w, ref_h))

    # Font setup for labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_thickness = 2
    text_color = (255, 255, 255)  # white

    # Infer cols if not provided
    if cols is None:
        cols = int(np.ceil(num_models / rows))

    # Total grid slots
    total_slots = rows * cols

    # Video frame size calculation
    frame_width = cols * ref_w + (cols - 1) * padding
    frame_height = rows * (ref_h + label_height) + (rows - 1) * padding

    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height)
    )

    for frame_idx in range(num_frames):
        # Start with blank canvas
        canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        for slot_idx in range(total_slots):
            row = slot_idx // cols
            col = slot_idx % cols

            # Top-left corner of this block
            x0 = col * (ref_w + padding)
            y0 = row * (ref_h + label_height + padding)

            block = np.zeros((ref_h + label_height, ref_w, 3), dtype=np.uint8)

            if slot_idx < num_models:
                img = resize(image_lists[slot_idx][frame_idx])
                block[label_height:, :, :] = img

                label = labels[slot_idx]
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

            # Paste block into canvas
            canvas[y0 : y0 + ref_h + label_height, x0 : x0 + ref_w, :] = block

        out.write(canvas)

    out.release()
    print(f"Video saved at {output_path}")


def main(args):
    folder_list = args.folder_list
    output_path = args.output_path
    fps = args.fps
    labels = args.labels
    padding = args.padding
    rows = args.rows
    cols = args.cols
    scale = args.scale

    image_lists = []
    for folder in folder_list:
        images = []
        for filename in sorted(os.listdir(folder)):
            image_path = os.path.join(folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image: {image_path}")
            if scale != 1.0:
                img = cv2.resize(
                    img,
                    (
                        int(img.shape[1] * scale),
                        int(img.shape[0] * scale),
                    ),
                )
            images.append(img)
        image_lists.append(images)

    images_to_video(
        image_lists,
        output_path,
        labels=labels,
        fps=fps,
        padding=padding,
        rows=rows,
        cols=cols,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize and compare depth maps from different models."
    )

    parser.add_argument(
        "--folder_list",
        type=str,
        nargs="+",
        required=True,
        help="Paths to folders that contain frames (one folder per model).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output video (e.g., 'comparison.mp4').",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        nargs="+",
        help="Labels for each model (must match number of folders).",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding between videos in pixels.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of rows in the grid.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Number of columns in the grid. If not set, it is inferred.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scaling factor for the output video size.",
    )

    args = parser.parse_args()
    main(args)
