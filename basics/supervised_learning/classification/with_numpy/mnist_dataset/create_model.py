import time
import random
import numpy as np
from os.path import join
from typing import Optional

from basics.supervised_learning.classification.with_numpy.activations import relu
from basics.supervised_learning.classification.with_numpy.activation_derivatives import relu_derivative
from basics.supervised_learning.classification.with_numpy.error_deltas import mse_error_delta
from basics.supervised_learning.classification.with_numpy.loss import mse_loss
from basics.supervised_learning.classification.with_numpy.mnist_dataset.mnist_dataloader import MnistDataloader
from basics.supervised_learning.classification.with_numpy.neural_network import NeuralNetwork

import matplotlib.pyplot as plt


# Config
INPUT_PATH: str = './archive'
TRAIN_SIZE: Optional[int] = None
TEST_SIZE: Optional[int] = None
NN_HIDDEN_SIZES: list[int] = [20, 20]
BATCH_SIZE: int = 30
LEARNING_RATE: float = 0.3
EPOCHS: int = 20


def main() -> None:
    # File paths
    training_images_filepath: str = join(INPUT_PATH, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath: str = join(INPUT_PATH, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath: str = join(INPUT_PATH, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath: str = join(INPUT_PATH, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Load dataset
    mnist_dataloader: MnistDataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Subsample if needed
    if TRAIN_SIZE is not None:
        x_train = x_train[:TRAIN_SIZE]
        y_train = y_train[:TRAIN_SIZE]
    if TEST_SIZE is not None:
        x_test = x_test[:TEST_SIZE]
        y_test = y_test[:TEST_SIZE]

    # Handle empty dataset
    if x_train.size == 0 or x_test.size == 0:
        print("No data to work with!")
        return

    # Flatten images
    num_train, rows, cols = x_train.shape
    num_test: int = x_test.shape[0]
    x_train_flat: np.ndarray = x_train.reshape(num_train, -1)
    x_test_flat: np.ndarray = x_test.reshape(num_test, -1)

    # Get NN sizes
    nn_input_size: int = x_train_flat.shape[1]
    nn_output_size: int = len(set(y_train.tolist()) | set(y_test.tolist()))

    # Init NN
    nn: NeuralNetwork = NeuralNetwork(
        input_size=nn_input_size,
        hidden_sizes=NN_HIDDEN_SIZES,
        output_size=nn_output_size,
        activation=relu,
        activation_derivative=relu_derivative,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        loss_fn=mse_loss,
        error_delta_fn=mse_error_delta
    )

    # Train
    nn.fit(x_train_flat, y_train, epochs=EPOCHS)

    # Test
    predictions: np.ndarray = nn.predict(x_test_flat)
    accuracy: float = np.mean(predictions == y_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Show sample predictions
    num_examples: int = 10
    sample_indices: list[int] = random.sample(range(len(x_test)), num_examples)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sample_indices):
        img: np.ndarray = x_test[idx]
        true_label: int = int(y_test[idx])
        pred_label: int = int(predictions[idx])

        plt.subplot(2, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Save model
    model_filename: str = f"./model{str(time.time()).replace('.', '')}.pkl"
    nn.save(model_filename)


if __name__ == "__main__":
    main()
