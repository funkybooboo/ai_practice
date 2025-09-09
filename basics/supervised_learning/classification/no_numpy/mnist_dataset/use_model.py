import os
import random
from os.path import join
from typing import List, Optional
import matplotlib.pyplot as plt

from basics.supervised_learning.classification.no_numpy.flatteners import flatten
from basics.supervised_learning.classification.no_numpy.mnist_dataset.mnist_dataloader import MnistDataloader
from basics.supervised_learning.classification.no_numpy.neural_network import NeuralNetwork

INPUT_PATH: str = './archive'
MODEL_PATH: Optional[str] = None
TEST_SIZE: Optional[int] = 1000


def get_most_recent_model_path(start_path: str) -> str:
    # List all files in the given directory
    files = [f for f in os.listdir(start_path) if f.endswith('.pkl')]
    if not files:
        raise FileNotFoundError(f"No .pkl files found in {start_path}")

    # Sort files based on the timestamp in the filename (numbers after 'model')
    files.sort(key=lambda f: int(f.split('model')[1].split('.pkl')[0]), reverse=True)

    # Return the most recent file
    most_recent_file = files[0]
    return join(start_path, most_recent_file)


def main() -> None:
    training_images_filepath: str = join(INPUT_PATH, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath: str = join(INPUT_PATH, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath: str = join(INPUT_PATH, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath: str = join(INPUT_PATH, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath
    )
    ( _, _ ), (test_images_raw, test_labels) = mnist_dataloader.load_data()

    # Optionally reduce test set size
    if TEST_SIZE is not None:
        test_images_raw = test_images_raw[:TEST_SIZE]
        test_labels = test_labels[:TEST_SIZE]

    test_images_flat: List[List[float]] = [flatten(images) for images in test_images_raw]

    model_path = MODEL_PATH
    if model_path is None:
        model_path = get_most_recent_model_path('./')

    nn: NeuralNetwork = NeuralNetwork.load(model_path)

    # Predict
    predictions: List[int] = nn.predict(test_images_flat)
    accuracy: float = sum(p == t for p, t in zip(predictions, test_labels)) / len(test_labels)
    print(f"Test Accuracy: {accuracy:.2%}")

    # Display some random test images
    num_examples = 10
    image_size = len(test_images_raw[0])
    sample_indices = random.sample(range(len(test_images_flat)), num_examples)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sample_indices):
        img_flat = test_images_flat[idx]
        img_2d = [img_flat[r * image_size:(r + 1) * image_size] for r in range(image_size)]
        true_label = test_labels[idx]
        pred_label = predictions[idx]

        plt.subplot(2, 5, i + 1)
        plt.imshow(img_2d, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
