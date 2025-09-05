from __future__ import annotations
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from typing import Tuple, List


# download dataset from here https://www.kaggle.com/datasets/hojjatk/mnist-dataset


class MnistDataloader:
    def __init__(
        self,
        training_images_filepath: str,
        training_labels_filepath: str,
        test_images_filepath: str,
        test_labels_filepath: str
    ) -> None:
        # Store file paths
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    @staticmethod
    def read_images_labels(
        images_filepath: str,
        labels_filepath: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Read labels from file
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels: np.ndarray = np.frombuffer(file.read(), dtype=np.uint8)

        # Read images from file
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data: np.ndarray = np.frombuffer(file.read(), dtype=np.uint8)

        # Normalize images to [0,1] and reshape
        images: np.ndarray = image_data.reshape(size, rows, cols).astype(np.float32) / 255.0

        return images, labels

    def load_data(
        self
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        # Load training and test sets
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


def show_images(images: np.ndarray, title_texts: List[str]) -> None:
    # Display images in a grid with titles
    cols: int = 5
    rows: int = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 8))
    for idx, (img, title) in enumerate(zip(images, title_texts)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()


def main() -> None:
    # Initialize dataloader
    input_path: str = './archive'
    mnist_dataloader: MnistDataloader = MnistDataloader(
        join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),
        join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'),
        join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'),
        join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    )

    # Load data
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Collect random images to show
    images_to_show: List[np.ndarray] = []
    titles_to_show: List[str] = []
    for _ in range(10):
        idx: int = random.randint(0, len(x_train) - 1)
        images_to_show.append(x_train[idx])
        titles_to_show.append(f"train[{idx}] = {y_train[idx]}")
    for _ in range(5):
        idx: int = random.randint(0, len(x_test) - 1)
        images_to_show.append(x_test[idx])
        titles_to_show.append(f"test[{idx}] = {y_test[idx]}")

    # Show images
    show_images(np.array(images_to_show), titles_to_show)


if __name__ == "__main__":
    main()
