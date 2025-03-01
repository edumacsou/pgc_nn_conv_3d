import os
import numpy as np
import warnings
import trimesh
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from trimesh import Trimesh
# from trimesh.viewer import SceneViewer
import open3d as o3d
from my_utils import load_mesh

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


def plot_3d_model(points: np.ndarray, title: str, save_path: str = None) -> None:
    """
    Plot a 3D model using matplotlib, ensuring it fits within a (1, 1, 1) cube.

    Args:
        points (np.ndarray): Array of points representing the 3D model.
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot image. If None, the plot is displayed.
    """
    fig = plt.figure(figsize=(10, 10))
    views = {
        "Isometrica": (35, 45, 120),
        "Lateral": (87, -135, -45),
        "Superior": (0, 90, 90),
        "Frontal": (0, 0, 90)
    }
    
    for i, (name, (elev, azim, roll)) in enumerate(views.items(), 1):
        ax = fig.add_subplot(2, 2, i, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_title(name)
        
        # Definir os limites dos eixos para garantir o cubo (1, 1, 1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # Forçar a mesma escala nos eixos
        ax.set_box_aspect([1, 1, 1])  # Aspecto do cubo

        # Configurações adicionais para melhor visualização
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Ajustar a visão para cada subplot
        ax.view_init(elev=elev, azim=azim, roll=roll)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        filename = title.split(":")[-1].strip().replace(" ", "_") + ".png"
        plt.savefig(f"{save_path + '/images'}/{filename}")
    else:
        plt.show()

points_amount = 10000

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