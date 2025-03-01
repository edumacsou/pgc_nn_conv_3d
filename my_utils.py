
#%%
import trimesh
import os
import numpy as np

## Load a mesh from a file
def load_mesh(filepath):
    """
    Load a mesh from a file. If the file contains a Scene, extract the first mesh.
    """
    # Load the file
    mesh_or_scene = trimesh.load(filepath)

    # Check if the loaded object is a Scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        print(f"File {filepath.split("/")[-1]} contains a Scene. Extracting the first mesh.")
        # Extract the first mesh from the Scene
        mesh = mesh_or_scene.geometry[list(mesh_or_scene.geometry.keys())[0]]
    else:
        print(f"File {filepath.split("/")[-1]} contains a single mesh.")
        mesh = mesh_or_scene

    return mesh

## Load meshes from .obj files given a folder_path
def load_meshes(folder_path: str):
    """
    Load 3D models from .obj files in a specified folder.

    Args:
        folder_path (str): Path to the folder containing .obj files.

    Returns:
        List[np.ndarray]: List of 3D models represented as arrays of vertices.
    """
    models = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".obj"):
            file_path = os.path.join(folder_path, filename)
            mesh = load_mesh(file_path)
            models[filename] = mesh

    return models

def load_pairings():
    """
    Load the pairings of models from the pairings.txt file.
    The file must contain lines in the format "input_filename,output_filename,time".
    Returns:
        List[Tuple[Tuple[str, int], str]]: List of tuples containing the input_filename,time and output_filename.
    """
    pairings = []
    with open("pairings.txt", "r") as f:
        for line in f:
            input_filename, output_filename, time = line.strip().split(",")
            pairings.append(((input_filename, int(time)), output_filename))

    return pairings

def load_data(points_amount = 1000):
    models_folder = f"/home/maciel/Documentos/Codes/TCC/pgc_nn_conv_3d/samples/sampled_models_{points_amount}"

    images = load_meshes(models_folder)
    pairs = load_pairings()

    X = [[images[X[0]].vertices.tolist(), X[1]] for X, _ in pairs]
    Y = [images[Y].vertices.tolist() for _, Y in pairs]

    return X, Y
#%%

if __name__ == "__main__":
    X, Y = load_data(1000)
    print(X)
    print(Y)