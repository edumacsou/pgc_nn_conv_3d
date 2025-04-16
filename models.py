from tensorflow.keras import layers, models, regularizers, Input
from typing import Tuple
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Input, Activation
from tensorflow.keras.constraints import max_norm
# from pytorch3d.loss import chamfer_distance

# Function to create a multi-input neural network model
def create_nn_v1(vertices_shape: Tuple[int, int]) -> models.Model:
    """
    Create a neural network model with two inputs:
    1. A (1000, 3) array for the vertices.
    2. An integer for the deformation amount.

    Args:
        vertices_shape (Tuple[int, int]): Shape of the vertices input (1000, 3).

    Returns:
        models.Model: Compiled neural network model.
    """
    # Input for the vertices
    vertices_input = Input(shape=vertices_shape, name="vertices_input")
    # Input for the deformation amount
    deformation_input = Input(shape=(1,), name="deformation_input")

    # Process the vertices input
    x = layers.Flatten()(vertices_input)
    x = layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(0.5)(x)

    # Process the deformation input
    y = layers.Dense(16, activation='relu')(deformation_input)
    y = layers.Dense(16, activation='relu')(y)

    # Concatenate the two branches
    combined = layers.concatenate([x, y])

    combined = layers.Dense(3**4, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(3**5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)

    # Output layer
    output = layers.Dense(np.prod(vertices_shape), activation='linear')(combined)
    output = layers.Reshape(vertices_shape)(output)

    # Create the model
    model = models.Model(inputs=[vertices_input, deformation_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model

def create_nn_v2(vertices_shape: Tuple[int, int]) -> models.Model:
    """
    Create a neural network model with two inputs:
    1. A (1000, 3) array for the vertices.
    2. An integer for the deformation amount.

    Args:
        vertices_shape (Tuple[int, int]): Shape of the vertices input (1000, 3).

    Returns:
        models.Model: Compiled neural network model.
    """
    # Input for the vertices
    vertices_input = Input(shape=vertices_shape, name="vertices_input")
    # Input for the deformation amount
    deformation_input = Input(shape=(1,), name="deformation_input")

    # Process the vertices input
    x = layers.Flatten()(vertices_input)
    y = layers.Flatten()(deformation_input)
    combined = layers.concatenate([x, y])

    combined = layers.Dense(3**6, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(3**5, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(combined)
    combined = layers.Dense(3**4, activation='relu', kernel_regularizer=regularizers.l2(0.01))(combined)
    combined = layers.Dropout(0.5)(x)
    
    combined = layers.concatenate([combined, x])

    # Output layer
    output = layers.Dense(np.prod(vertices_shape), activation='linear')(combined)
    output = layers.Reshape(vertices_shape)(output)

    # Create the model
    model = models.Model(inputs=[vertices_input, deformation_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    return model


def create_nn_v3(shape: Tuple[int, int]) -> models.Model:
    pass
#     model = Sequential([
#         Input(shape=(shape,1)),  # Camada de entrada
#         Dense(512),  # Camada densa
#         BatchNormalization(),  # Batch Normalization
#         Activation('relu'),  # Ativação ReLU
#         Dense(256),  # Camada densa
#         BatchNormalization(),  # Batch Normalization
#         Activation('relu'),  # Ativação ReLU
#         Dense(shape, activation='linear')  # Camada de saída
#     ])

#     # Compilar o modelo com Chamfer Loss
#     model.compile(optimizer='adam', loss=chamfer_distance)

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, ReLU, GlobalMaxPooling1D, Concatenate, Lambda
from tensorflow.keras.models import Model


def build_pointnet_cnn(input_shape=(2000, 3)):
    """
    Versão melhorada com:
    - Normalização rigorosa
    - Regularização reforçada
    - Controle de escala na saída
    """
    # Inputs
    vertices_input = layers.Input(shape=input_shape, name="vertices_input")
    deformation_input = layers.Input(shape=(1,), name="deformation_input")
    
    # Normalização dos inputs
    normalized_verts = layers.Lambda(lambda x: x / 1000.)(vertices_input)  # Assume valores originais ~0-1
    normalized_def = layers.Lambda(lambda x: x / 100.)(deformation_input)  # Normaliza tempo 2-37 para ~0.02-0.37

    # Combine inputs
    def expand_and_concat(inputs):
        verts, deform = inputs
        deform_exp = tf.tile(deform[:, tf.newaxis, :], [1, tf.shape(verts)[1], 1])
        return tf.concat([verts, deform_exp], axis=-1)
    
    combined = layers.Lambda(expand_and_concat)([normalized_verts, normalized_def])

    # Processamento principal com regularização
    x = layers.Conv1D(128, 1, kernel_regularizer=regularizers.l2(0.01),
                     kernel_constraint=max_norm(3))(combined)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Conv1D(256, 1, kernel_regularizer=regularizers.l2(0.01),
                     kernel_constraint=max_norm(3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    # Agregação global
    global_feat = layers.GlobalMaxPooling1D()(x)
    global_feat = layers.Dense(128, activation='relu')(global_feat)
    
    # Combinação com features locais
    global_exp = layers.RepeatVector(input_shape[0])(global_feat)
    x = layers.Concatenate()([combined, global_exp])
    
    # Camadas finais
    x = layers.Conv1D(256, 1, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    
    # Saída com ativação tanh para limitar valores (-1 a 1)
    output = layers.Conv1D(3, 1, activation='tanh')(x)
    
    # Escala final para ajustar à faixa desejada
    final_output = layers.Lambda(lambda x: x * 1000.)(output)  # Reverte a normalização

    model = models.Model(inputs=[vertices_input, deformation_input], outputs=final_output)
    
    # Otimizador com learning rate ajustado
    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss='mse')
    
    return model