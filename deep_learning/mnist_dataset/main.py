from deep_learning.activation_derivatives import sigmoid_derivative
from deep_learning.activations import sigmoid
from deep_learning.flatteners import flatten
from deep_learning.mnist_dataset.mnist_dataloader import MnistDataloader
from os.path import join
import random
import matplotlib.pyplot as plt
from typing import List, Optional

from deep_learning.neural_network import NeuralNetwork
from deep_learning.normalizers import normalize_image


TRAIN_SIZE: Optional[int] = 5000        # number of training samples to use, None for all
TEST_SIZE: Optional[int] = 1000         # number of test samples to use, None for all
IMAGE_SIZE: int = 28                    # height/width of MNIST images
NN_HIDDEN_SIZES: List[int] = [128, 64]  # list of hidden layer sizes, can add more layers
NN_OUTPUT_SIZE: int = 10                # number of classes
LR: float = 0.1                         # learning rate
EPOCHS: int = 5                         # number of epochs
BATCH_SIZE: int = 32                    # number of batches
INPUT_PATH: str = './archive'           # folder path for MNIST data


def main() -> None:
    training_images_filepath: str = join(INPUT_PATH, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath: str = join(INPUT_PATH, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath: str = join(INPUT_PATH, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath: str = join(INPUT_PATH, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader: MnistDataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath
    )

    # Load raw image data
    (train_images_raw, train_labels), (test_images_raw, test_labels) = mnist_dataloader.load_data()

    # Optionally reduce dataset sizes
    if TRAIN_SIZE is not None:
        train_images_raw = train_images_raw[:TRAIN_SIZE]
        train_labels = train_labels[:TRAIN_SIZE]
    if TEST_SIZE is not None:
        test_images_raw = test_images_raw[:TEST_SIZE]
        test_labels = test_labels[:TEST_SIZE]

    # Flatten and normalize images
    train_images_flat: List[List[float]] = [normalize_image(flatten(img)) for img in train_images_raw]
    test_images_flat: List[List[float]] = [normalize_image(flatten(img)) for img in test_images_raw]

    # Dynamically calculate the input size (flattened image dimensions)
    NN_INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE  # 28x28 MNIST images

    # Build MLP with dynamic hidden layers
    nn: NeuralNetwork = NeuralNetwork(
        input_size=NN_INPUT_SIZE,
        hidden_sizes=NN_HIDDEN_SIZES,
        output_size=NN_OUTPUT_SIZE,
        activation=sigmoid,
        activation_derivative=sigmoid_derivative,
        lr=LR,
        batch_size=BATCH_SIZE,
    )

    # Train
    nn.fit(train_images_flat, train_labels, epochs=EPOCHS)

    # Evaluate
    predictions: List[float] = nn.predict(test_images_flat)
    accuracy: float = sum(p == t for p, t in zip(predictions, test_labels)) / len(test_labels)
    print(f"Test Accuracy: {accuracy:.2%}")

    # Display some random test images
    num_examples: int = 10
    sample_indices: List[int] = random.sample(range(len(test_images_flat)), num_examples)
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(sample_indices):
        img_flat: List[float] = test_images_flat[idx]
        img_2d: List[List[float]] = [img_flat[r * IMAGE_SIZE:(r + 1) * IMAGE_SIZE] for r in range(IMAGE_SIZE)]
        true_label: float = test_labels[idx]
        pred_label: float = predictions[idx]

        plt.subplot(2, 5, i + 1)
        plt.imshow(img_2d, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
