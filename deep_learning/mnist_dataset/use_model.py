import random
from os.path import join
from typing import List, Optional
import matplotlib.pyplot as plt

from deep_learning.flatteners import flatten
from deep_learning.mnist_dataset.mnist_dataloader import MnistDataloader
from deep_learning.neural_network import NeuralNetwork

# Config
INPUT_PATH: str = './archive'
MODEL_PATH: str = './model17565163193815045.pkl'
TEST_SIZE: Optional[int] = 1000

def main() -> None:
    # File paths
    training_images_filepath: str = join(INPUT_PATH, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath: str = join(INPUT_PATH, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath: str = join(INPUT_PATH, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath: str = join(INPUT_PATH, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    # Load MNIST data
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

    # Flatten and normalize images
    test_images_flat: List[List[float]] = [flatten(img) for img in test_images_raw]

    # Load saved model
    nn = NeuralNetwork.load(MODEL_PATH)

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
