import argparse
import o3d
import os


def main(args):
    mesh_path = args.mesh_path
    assert os.path.exists(mesh_path), f"Mesh path does not exist: {mesh_path}"

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    o3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize the output of the model")
    parser.add_arugment("--mesh_path", type=str, help="Path to the mesh file")
    args = parser.parse_args()
    main(args)
