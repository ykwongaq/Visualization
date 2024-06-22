import argparse
import os
import cv2
import numpy as np
import open3d as o3d


def read_depth(depth_path):
    if depth_path.endswith(".npy"):
        depth = np.load(depth_path)
    elif depth_path.endswith(".png"):
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError(f"Unsupported depth format: {depth_path}")
    return depth


def denormalize_depth(depth, min_depth, max_depth):
    return depth * (max_depth - min_depth) + min_depth


def main(args):

    image_path = args.image_path
    assert os.path.exists(image_path), f"Image path does not exist: {image_path}"

    depth_path = args.depth_path
    assert os.path.exists(depth_path), f"Depth path does not exist: {depth_path}"

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    min_depth = args.min_depth
    max_depth = args.max_depth

    assert (
        min_depth < max_depth
    ), f"Minimum depth must be less than maximum depth. Got: {min_depth} and {max_depth}"
    assert min_depth > 0, f"Minimum depth must be greater than zero. Got: {min_depth}"

    image = cv2.imread(image_path)
    depth = read_depth(depth_path)
    print(f"{depth.min()}, {depth.max()}")
    depth = denormalize_depth(depth, min_depth, max_depth)

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    height, width = depth.shape
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))

    x = xx.flatten()
    y = yy.flatten()
    z = depth.flatten()
    colors = image.reshape(-1, 3) / 255.0

    # Remove points with zero depth
    print(z)
    valid_points = z > 0
    print(f"Number of valid points: {np.sum(valid_points)}")
    x = x[valid_points]
    y = y[valid_points]
    z = z[valid_points]
    colors = colors[valid_points]

    # Create point cloud
    points = np.vstack((x, y, z)).T
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([point_cloud])

    pass


if __name__ == "__main__":
    DEFAULT_OUTPUT_FOLDER = "./output"
    DEFAULT_IMAGE_FILE = "images/20180915_UTP_QUADRAT_SOCBOR1_40M_1_cropped_image.png"
    DEFAULT_DEPTH_FILE = "depths/20180915_UTP_QUADRAT_SOCBOR1_40M_1_depth.npy"
    DEFAULT_MIN_DEPTH = 1
    DEFAULT_MAX_DEPTH = 1000

    parser = argparse.ArgumentParser(description="Visualize the output of the model")
    parser.add_argument(
        "--image_path",
        type=str,
        default=DEFAULT_IMAGE_FILE,
        help=f"Path to the input image. Default: {DEFAULT_IMAGE_FILE}",
    )
    parser.add_argument(
        "--depth_path",
        type=str,
        default=DEFAULT_DEPTH_FILE,
        help=f"Path to the depth map. Default: {DEFAULT_DEPTH_FILE}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Path to save the visualization. Default: {DEFAULT_OUTPUT_FOLDER}",
    )
    parser.add_argument(
        "--min_depth",
        type=int,
        default=DEFAULT_MIN_DEPTH,
        help=f"Minimum depth value to visualize. Default: {DEFAULT_MIN_DEPTH}",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"Maximum depth value to visualize. Default: {DEFAULT_MAX_DEPTH}",
    )
    args = parser.parse_args()
    main(args)
