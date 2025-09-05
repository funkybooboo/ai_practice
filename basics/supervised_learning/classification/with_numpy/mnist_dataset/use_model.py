import os
import random
from os.path import join
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from basics.supervised_learning.classification.with_numpy.mnist_dataset.mnist_dataloader import MnistDataloader
from basics.supervised_learning.classification.with_numpy.neural_network import NeuralNetwork


# Config
INPUT_PATH: str = './archive'
MODEL_PATH: Optional[str] = None
TEST_SIZE: Optional[int] = 1000


def get_most_recent_model_path(start_path: str) -> str:
    # List all .pkl files in the directory
    files: list[str] = [f for f in os.listdir(start_path) if f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {start_path}")

    # Sort files by numeric timestamp in filename, newest first
    files.sort(key=lambda f: int(f.split('model')[1].split('.pkl')[0]), reverse=True)
    return join(start_path, files[0])


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
    (_, _), (x_test, y_test) = mnist_dataloader.load_data()

    # Reduce test set if specified
    if TEST_SIZE is not None:
        x_test = x_test[:TEST_SIZE]
        y_test = y_test[:TEST_SIZE]

    # Flatten test images
    num_test, rows, cols = x_test.shape
    x_test_flat: np.ndarray = x_test.reshape(num_test, -1)

    # Load the most recent model if none specified
    model_path: str = MODEL_PATH or get_most_recent_model_path('./')
    nn: NeuralNetwork = NeuralNetwork.load(model_path)

    # Predict and compute accuracy
    predictions: np.ndarray = nn.predict(x_test_flat)
    accuracy: float = np.mean(predictions == y_test) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Display sample predictions
    num_examples: int = 10
    sample_indices: list[int] = random.sample(range(len(x_test)), num_examples)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sample_indices):
        img: np.ndarray = x_test[idx]
        true_label: int = int(y_test[idx])
        pred_label: int = int(predictions[idx])

        plt.subplot(2, 5, i + 1)
        plt.imshow(img, cmap='gray')  # Show image
        plt.title(f"True: {true_label}\nPred: {pred_label}")  # Show labels
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
