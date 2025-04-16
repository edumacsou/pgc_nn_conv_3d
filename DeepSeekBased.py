import os
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Or 'Agg' if you don't need a graphical interface
import matplotlib.pyplot as plt
import tensorflow as tf
from my_utils import load_data, MODELS_FOLDER, plot_3d_model
from models import create_nn_v1, create_nn_v2, create_nn_v3, build_pointnet_cnn

def main(model_creator):
    # Load data
    points_amount = 2000
    X, Y = load_data(points_amount)

    # Prepare the inputs
    # X is a list of 19 elements, where each element is [vertices, deformation_amount]
    vertices = np.array([x[0] for x in X])  # Shape: (19, 1000, 3)
    deformation_amounts = np.array([x[1] for x in X]).reshape(-1, 1)  # Shape: (19, 1)

    # Convert Y to a numpy array
    Y = np.array(Y)  # Shape: (19, 1000, 3)


    # Verificação dos dados
    print("\n=== Verificação dos Dados ===")
    print(f"Vértices: {vertices.shape}, Valores: {vertices.min():.2f} to {vertices.max():.2f}")
    print(f"Deformação: {deformation_amounts.shape}, Valores: {deformation_amounts.min()} to {deformation_amounts.max()}")
    print(f"Target Y: {Y.shape}, Valores: {Y.min():.2f} to {Y.max():.2f}")

        
    # Verificação adicional dos dados
    print("\n=== Estatísticas dos Dados ===")
    print(f"Média vértices: {np.mean(vertices):.4f} ± {np.std(vertices):.4f}")
    print(f"Média targets: {np.mean(Y):.4f} ± {np.std(Y):.4f}")
    
    # Callbacks adicionais
    # early_stop = tf.keras.callbacks.EarlyStopping(
    #     patience=13, 
    #     restore_best_weights=True,
    #     monitor='val_loss'
    # )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=0.5,
        patience=5,
        min_lr=1e-6
    )


    # Visualização dos dados brutos
    plot_3d_model(vertices[0], "Input Sample")
    plot_3d_model(Y[0], "Target Sample")



    # Create and compile the model
    input_shape = (points_amount, 3)
    model = model_creator(input_shape)
    model.summary()  # <-- Mostra arquitetura

    # Train the model
    # hist = model.fit([vertices, deformation_amounts], Y, epochs=50, batch_size=1)
        # Treinamento com mais épocas
    hist = model.fit(
        [vertices, deformation_amounts],
        Y,
        epochs=200,  # Aumentado
        batch_size=4,
        validation_split=0.2,
        callbacks=[reduce_lr],
        verbose=2
    )

    # Test the model
    test_vertices = vertices[0:1]  # First skull's vertices
    test_deformation = deformation_amounts[0:1]  # First skull's deformation amount
    predicted_output = model.predict([test_vertices, test_deformation])

    print("\n=== Predição ===")
    print(f"Predição shape: {predicted_output[0].shape}")
    print(f"Valores preditos: min={predicted_output[0].min():.2f}, max={predicted_output[0].max():.2f}")
    print(f"Target real: min={Y[0].min():.2f}, max={Y[0].max():.2f}")
    


    ## Save trained model on MODEL_FOLDER
    if not os.path.exists(os.path.join(MODELS_FOLDER,f"sampled_models_{points_amount}")):
        os.makedirs(os.path.join(MODELS_FOLDER,f"sampled_models_{points_amount}"))

    # Save the model to disk
    model.save(os.path.join(MODELS_FOLDER,f"sampled_models_{points_amount}", f"trained_model_{points_amount}.keras"))

    # Save the model's architecture to disk
    with open(os.path.join(MODELS_FOLDER,f"sampled_models_{points_amount}", f"trained_model_{points_amount}_architecture.json"), "w") as f:
        f.write(model.to_json())

    try:
        # Save the model's loss to disk
        with open(os.path.join(MODELS_FOLDER,f"sampled_models_{points_amount}", f"trained_model_{points_amount}_errors.txt"), "w") as f:
            f.write(str(hist.history['loss']))

        # Visualize the loss
        plt.plot(hist.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
    except Exception as e:
        print("Error: ",e)

    # Visualize the input model
    # plot_3d_model(test_vertices[0], "Input Model")
    # plot_3d_model(Y[0], "Train Model")

    # Visualize the predicted output model
    plot_3d_model(predicted_output[0], "Predicted Output Model")

if __name__ == "__main__":
    main(create_nn_v1)
    # main(create_nn_v2)
    # main(create_nn_v3)
    # main(build_pointnet_cnn)