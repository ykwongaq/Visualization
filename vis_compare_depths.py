import matplotlib.pyplot as plt
import os
import argparse

from PIL import Image
from tqdm import tqdm


def is_image(image_path:str):
    try:
        img = Image.open(image_path)
        img.verify()
        return True
    except:
        return False
    
def main(args):
    
    input_dirs = args.input_dirs
    captions = args.captions
    assert len(input_dirs) == len(captions), f"Number of captions do not match the number of folders. Expected {len(input_dirs)} captions, got {len(captions)}"

    # Dimension of the layout
    num_rows = args.num_rows
    num_cols = args.num_cols
    assert num_rows*num_cols >= len(input_dirs), f"Number of rows and columns should be greater than the number of folders. Expected {len(input_dirs)} <= {num_rows*num_cols}"

    # Extract images files)
    images_files_list = []
    for folder in args.input_dirs:
        assert os.path.exists(folder), f"Folder {folder} does not exist"
        print(f"Collecting images from {folder}")
        images_files = []
        for image_filename in os.listdir(folder):
            image_file = os.path.join(folder, image_filename)
            if is_image(image_file):
                images_files.append(image_file)
        images_files_list.append(images_files)
    
    # Assert that the number of images is the same
    num_images = len(images_files_list[0])
    for images_files in images_files_list:
        assert len(images_files) == num_images, "Number of images in the folders do not match"

    # Sort the images
    images_files_list = [sorted(images_files) for images_files in images_files_list]
    print(f"Collected {num_images} images from each folder")

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Create the visualization

    for image_idx in tqdm(range(num_images), desc="Creating visualization ..."):
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
        axs_flat = axs.flatten()
        filename = None
        for folder_idx, images_files in enumerate(images_files_list):
            image_filename = images_files[image_idx]
            filename = os.path.basename(image_filename)
            image = Image.open(image_filename)
            axs_flat[folder_idx].imshow(image)
            axs_flat[folder_idx].axis("off")
            axs_flat[folder_idx].set_title(captions[folder_idx])
        plt.tight_layout()
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path)
        plt.close(fig)
    
    print(f"All visualization have been saved to {output_dir}")
    pass

if __name__ == "__main__":
    print("Make sure that the corresponding depth maps have the same name or are in the same order after sorting by name")

    parser = argparse.ArgumentParser(description='Visualize the depth maps')
    parser.add_argument("--input_dirs", nargs="+", type=str, help="List of directories containing depth maps")
    parser.add_argument("--output_dir", type=str, help="Output directory to save the visualization")
    parser.add_argument("--num_rows", type=int, default=1, help="Number of rows in the visualization")
    parser.add_argument("--num_cols", type=int, default=1, help="Number of columns in the visualization")
    parser.add_argument("--captions", nargs="+", type=str, help="List of captions for each depth map")
    args = parser.parse_args()
    main(args)