#%%
import trimesh
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

MODELS_FOLDER = f"/home/maciel/Documentos/Codes/TCC/pgc_nn_conv_3d/samples/"

## Load a mesh from a file
def load_mesh(filepath):
    """
    Load a mesh from a file. If the file contains a Scene, extract the first mesh.
    """
    # Load the file
    mesh_or_scene = trimesh.load(filepath)

    # Check if the loaded object is a Scene
    if isinstance(mesh_or_scene, trimesh.Scene):
        print(f"File {filepath.split('/')[-1]} contains a Scene. Extracting the first mesh.")
        # Extract the first mesh from the Scene
        mesh = mesh_or_scene.geometry[list(mesh_or_scene.geometry.keys())[0]]
    else:
        print(f"File {filepath.split('/')[-1]} contains a single mesh.")
        mesh = mesh_or_scene

    return mesh

## Load meshes from .obj files given a folder_path
def load_meshes(folder_path: str):
    """
    Load 3D models from .obj files in a specified folder.

    Args:
        folder_path (str): Path to the folder containing .obj files.

    Returns:
        Dict[str, List[List[float]]]: Dictionary where keys are filenames and values are lists of vertices (as lists of floats).
    """
    models = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".obj"):
            file_path = os.path.join(folder_path, filename)
            mesh = load_mesh(file_path)
            # Convert vertices to a list of lists of floats
            models[filename] = mesh.vertices.tolist()

    return models

# Função para reindexar os pontos
def reindex_points(X):
    X_reindexed = np.zeros_like(X)

    sorting_key = X[:, 0] + X[:, 1] + X[:, 2]
    sorted_indices = np.argsort(sorting_key)
    
    return X[sorted_indices]


def load_pairings():
    """
    Load the pairings of models from the pairings.txt file.
    The file must contain lines in the format "input_filename,output_filename,time".
    Returns:
        List[Tuple[Tuple[str, int], str]]: List of tuples containing the input_filename, time, and output_filename.
    """
    pairings = []
    with open("pairings.txt", "r") as f:
        for line in f:
            input_filename, output_filename, time = line.strip().split(",")
            pairings.append(((input_filename, int(time)), output_filename))

    return pairings

def load_data(points_amount=1000):
    """
    Load data for training the model.

    Args:
        points_amount (int): Number of points to sample from each model.

    Returns:
        Tuple[List[List[List[float]]], List[List[List[float]]]: X and Y data.
            X is a list of tuples, where each tuple contains:
                - A list of vertices (as lists of floats).
                - An integer representing the deformation amount.
            Y is a list of lists of vertices (as lists of floats).
    """

    models_folder = os.path.join(MODELS_FOLDER,f"sampled_models_{points_amount}")

    # Load meshes and convert vertices to lists of lists of floats
    images = load_meshes(models_folder)
    reindexed_images = {k: reindex_points(np.array(v)) for k, v in images.items()}
    pairs = load_pairings()

    # Prepare X and Y data
    X = [[reindexed_images[X_aux[0]], X_aux[1]] for X_aux, _ in pairs]  # X is a list of [vertices, deformation_amount]
    Y = [reindexed_images[Y_aux] for _, Y_aux in pairs]  # Y is a list of vertices

    return X, Y


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

#%%

if __name__ == "__main__":
    X, Y = load_data(1000)
    print("X (input data):")
    print(type(X))
    print(type(X[0]), len(X[0]), type(X[1]))
    print(type(X[0][0]), len(X[0][0]))
    print(type(X[0][0][0]), len(X[0][0][0]), X[0][0][0])
    print(type(X[0][0][0][0]), X[0][0][0][0])
    # print("Y (output data):")
    # print(Y)