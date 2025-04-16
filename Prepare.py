import os
import numpy as np
import warnings
import trimesh
# from trimesh import Trimesh
# from trimesh.viewer import SceneViewer
import open3d as o3d
from my_utils import load_mesh, plot_3d_model

# Suppress warnings from trimesh
warnings.filterwarnings("ignore")
# Set Open3D verbosity level to suppress warnings
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# Function to normalize points to the range [0, 1] without deformation
def normalize_points(points: np.ndarray) -> np.ndarray:
    """
    Normalize points to the range [0, 1] without deformation.
    """
    min_vals = np.min(points, axis=0)
    normalized_points = (points - min_vals)
    max_val = np.max(normalized_points)
    normalized_points = normalized_points / max_val
    return normalized_points


points_amount = 1000

# Folder containing the .obj files
input_folder = "/home/maciel/Documentos/Codes/TCC/pgc_nn_conv_3d/samples/public_models"
output_folder = f"/home/maciel/Documentos/Codes/TCC/pgc_nn_conv_3d/samples/sampled_models_{points_amount}"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_folder + '/images'):
    os.makedirs(output_folder + '/images')

# Iterate over all .obj files in the input folder
for filename in os.listdir(input_folder):
# for filename in ['homo-erectus-skull.obj']:
    if filename.endswith(".obj"):
        # Load the mesh
        filepath = os.path.join(input_folder, filename)
        mesh = load_mesh(filepath)

        # Sample 1000 points from the mesh surface
        sampled_points, _ = trimesh.sample.sample_surface(mesh, count=points_amount)

        # Normalize the vertices
        sampled_points = normalize_points(sampled_points)

        #### V1: Trimesh Viewer not Working
        # # Create a new mesh from the sampled points
        # sampled_mesh = Trimesh(vertices=sampled_points)

        # # Display the resulting mesh
        # scene = trimesh.Scene(sampled_mesh)
        # SceneViewer(scene)

        # # Save the sampled mesh to the output folder
        # output_filepath = os.path.join(output_folder, f"sampled_{filename}")
        # sampled_mesh.export(output_filepath)

        # print(f"Processed and saved: {output_filepath}")

        #### V2:
        # Create an Open3D point cloud from the sampled points
        # point_cloud = o3d.geometry.PointCloud()
        # point_cloud.points = o3d.utility.Vector3dVector(sampled_points)

        # # Visualize the point cloud using Open3D
        # o3d.visualization.draw_geometries([point_cloud], window_name=f"Sampled Points: {filename}")

        #### V3 : Ploting with matplotLib
        plot_3d_model(sampled_points, f"Sampled Points: {filename}", save_path=output_folder)

        # Save the sampled points as a new .obj file
        output_filepath = os.path.join(output_folder, f"sampled_{filename}")
        with open(output_filepath, "w") as f:
            for point in sampled_points:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")

        print(f"Processed and saved: {filename}")

print("All files processed.")

