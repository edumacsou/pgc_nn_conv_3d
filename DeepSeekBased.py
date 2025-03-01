import os
import numpy as np
import trimesh
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras import layers, models, regularizers
from typing import List, Tuple
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Agg' se você não precisar de uma interface gráfica
import matplotlib.pyplot as plt
from my_utils import load_data


# Função para visualizar modelos 3D
def plot_3d_model(points: np.ndarray, title: str) -> None:
    """
    Plot a 3D model using matplotlib.

    Args:
        points (np.ndarray): Array of points representing the 3D model.
        title (str): Title of the plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)

    # Definir a mesma escala para todos os eixos
    max_range = np.array([points[:, 0].max() - points[:, 0].min(), 
                          points[:, 1].max() - points[:, 1].min(), 
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()



####################################


# # Caminho para a pasta com os modelos .obj
# folder_path = "./samples"

# # Carregar modelos
# models = load_models_from_obj(folder_path)

# Normalizar pontos (escolha um número fixo de pontos, por exemplo, 1000)
points_amount = 1000
# normalized_models = normalize_points(models, target_points)

# Preparar X e Y
# X, Y = prepare_data(normalized_models)

# Definir o modelo da rede neural

# def create_simple_nn(input_shape: Tuple[int, int]) -> models.Sequential:
def create_simple_nn(input_shape) -> models.Sequential:
    """
    Create a simple neural network model.

    Args:
        input_shape (Tuple[int, int]): Shape of the input data.

    Returns:
        models.Sequential: Compiled neural network model.
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(np.prod(input_shape), activation='linear'),
        layers.Reshape(input_shape)
    ])
    return model



# Criar e compilar o modelo
input_shape = (points_amount, 3, 1)
model = create_simple_nn(input_shape)
model.compile(optimizer='adam', loss='mse')


models_folder = f"/home/maciel/Documentos/Codes/TCC/pgc_nn_conv_3d/samples/sampled_models_{points_amount}"

X, Y = load_data(points_amount)

print(X)
print(Y)
# Treinar o modelo
model.fit(X, Y, epochs=50, validation_split=0.2)

# Testar o modelo
test_input = X[0:1]  # Primeiro crânio como teste
predicted_output = model.predict(test_input)

# Visualizar o modelo de entrada
plot_3d_model(test_input[0], "Input Model")
plot_3d_model(Y[0:1][0], "Train Model")

# Visualizar o modelo de saída (predição)
plot_3d_model(predicted_output[0], "Predicted Output Model")

# %%
