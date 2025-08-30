import time
from os.path import join
import random
from typing import List, Optional

from deep_learning.activation_derivatives import relu_derivative
from deep_learning.activations import relu
from deep_learning.flatteners import flatten
from deep_learning.mnist_dataset.mnist_dataloader import MnistDataloader
from deep_learning.neural_network import NeuralNetwork

import matplotlib.pyplot as plt

# things to not really touch
INPUT_PATH: str = './archive'

# things to touch
TRAIN_SIZE: Optional[int] = 10000
TEST_SIZE: Optional[int] = 2000
NN_HIDDEN_SIZES: List[int] = [16, 16]
BATCH_SIZE: int = 50
LR = 0.001
EPOCHS = 10

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

    if len(train_labels) == 0 or \
        len(test_labels) == 0 or \
        len(train_images_raw) == 0 or \
        len(train_images_raw[0]) == 0 or \
        len(test_images_raw) == 0 or \
        len(test_images_raw[0]) == 0:
        print("No data to work with!")
        return

    # Flatten and normalize images
    train_images_flat: List[List[float]] = [flatten(img) for img in train_images_raw]
    test_images_flat: List[List[float]] = [flatten(img) for img in test_images_raw]

    image_size: int = len(train_images_raw[0])
    nn_input_size: int = len(train_images_flat[0])
    nn_output_size: int = len(set(train_labels).union(set(test_labels)))

    # Build MLP with dynamic hidden layers
    nn: NeuralNetwork = NeuralNetwork(
        input_size=nn_input_size,
        hidden_sizes=NN_HIDDEN_SIZES,
        output_size=nn_output_size,
        activation=relu,
        activation_derivative=relu_derivative,
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
        img_2d: List[List[float]] = [img_flat[r * image_size:(r + 1) * image_size] for r in range(image_size)]
        true_label: float = test_labels[idx]
        pred_label: float = predictions[idx]

        plt.subplot(2, 5, i + 1)
        plt.imshow(img_2d, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    nn.save(f"./model{str(time.time()).replace('.', '')}.pkl")


if __name__ == "__main__":
    main()
