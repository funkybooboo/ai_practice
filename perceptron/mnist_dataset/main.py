from perceptron.flatteners import flatten
from perceptron.mnist_dataset.mnist_dataloader import MnistDataloader
from os.path import join
import random
import matplotlib.pyplot as plt
from typing import List, Optional

from perceptron.mlp import Mlp
from perceptron.normalizers import normalize_image


TRAIN_SIZE: Optional[int] = 100       # number of training samples to use, None for all
TEST_SIZE: Optional[int] = 10        # number of test samples to use, None for all
NUM_EXAMPLES: int = 10                 # number of test images to display
MLP_INPUT_SIZE: int = 784              # flattened 28x28 MNIST images
MLP_HIDDEN_SIZE: int = 20              # hidden layer size
MLP_OUTPUT_SIZE: int = 10              # number of classes
LR: float = 0.1                        # learning rate
EPOCHS: int = 5                        # number of epochs
INPUT_PATH: str = './archive'          # the folder path that holds all the mnist data files
IMAGE_SIZE: int = 28                   # height/width of MNIST images


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
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Optionally reduce dataset sizes
    if TRAIN_SIZE is not None:
        x_train = x_train[:TRAIN_SIZE]
        y_train = y_train[:TRAIN_SIZE]
    if TEST_SIZE is not None:
        x_test = x_test[:TEST_SIZE]
        y_test = y_test[:TEST_SIZE]

    # Flatten images
    X_train_flat: List[List[float]] = [flatten(img) for img in x_train]
    X_test_flat: List[List[float]] = [flatten(img) for img in x_test]

    # Normalize
    X_train: List[List[float]] = [normalize_image(img) for img in X_train_flat]
    X_test: List[List[float]] = [normalize_image(img) for img in X_test_flat]

    # Build MLP
    mlp: Mlp = Mlp(
        input_size=MLP_INPUT_SIZE,
        hidden_size=MLP_HIDDEN_SIZE,
        output_size=MLP_OUTPUT_SIZE
    )

    # Train
    mlp.fit(X_train, y_train, lr=LR, epochs=EPOCHS)

    # Evaluate
    preds: List[int] = mlp.predict(X_test)
    acc: float = sum(p == t for p, t in zip(preds, y_test)) / len(y_test)
    print(f"Test accuracy: {acc:.2%}")

    # Pick some random test images
    indices: List[int] = random.sample(range(len(X_test)), NUM_EXAMPLES)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        img: List[float] = X_test[idx]
        # reshape 1D list back to 28x28 2D list
        img_2d: List[List[float]] = [img[r*IMAGE_SIZE:(r+1)*IMAGE_SIZE] for r in range(IMAGE_SIZE)]
        true_label: int = y_test[idx]
        pred_label: int = preds[idx]

        plt.subplot(2, 5, i + 1)
        plt.imshow(img_2d, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
