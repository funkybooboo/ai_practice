from __future__ import annotations
import numpy as np
import struct
from array import array
import random
import matplotlib.pyplot as plt
from os.path import join
from typing import List, Tuple, Union


#
# MNIST Data Loader Class
#
class MnistDataloader:
    def __init__(
        self,
        training_images_filepath: str,
        training_labels_filepath: str,
        test_images_filepath: str,
        test_labels_filepath: str
    ) -> None:
        self.training_images_filepath: str = training_images_filepath
        self.training_labels_filepath: str = training_labels_filepath
        self.test_images_filepath: str = test_images_filepath
        self.test_labels_filepath: str = test_labels_filepath

    @staticmethod
    def read_images_labels(
        images_filepath: str,
        labels_filepath: str
    ) -> Tuple[List[np.ndarray], array]:
        labels: array
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(
                    f'Magic number mismatch, expected 2049, got {magic}'
                )
            labels = array("B", file.read())

        image_data: array
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(
                    f'Magic number mismatch, expected 2051, got {magic}'
                )
            image_data = array("B", file.read())

        images: List[np.ndarray] = []
        for _ in range(size):
            images.append(np.zeros((rows, cols), dtype=np.uint8))

        for i in range(size):
            img: np.ndarray = np.array(
                image_data[i * rows * cols:(i + 1) * rows * cols],
                dtype=np.uint8
            ).reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(
        self
    ) -> Tuple[Tuple[List[np.ndarray], array], Tuple[List[np.ndarray], array]]:
        x_train, y_train = self.read_images_labels(
            self.training_images_filepath, self.training_labels_filepath
        )
        x_test, y_test = self.read_images_labels(
            self.test_images_filepath, self.test_labels_filepath
        )
        return (x_train, y_train), (x_test, y_test)


#
# Helper function to show a list of images with their relating titles
#
def show_images(images: List[np.ndarray], title_texts: List[str]) -> None:
    cols: int = 5
    rows: int = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index: int = 1
    for image, title_text in zip(images, title_texts):
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if title_text != '':
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()


def main():
    #
    # Set file paths based on added MNIST Datasets
    #
    input_path: str = './archive'
    training_images_filepath: str = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath: str = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath: str = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath: str = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    #
    # Load MINST dataset
    #
    mnist_dataloader: MnistDataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images
    #
    images_2_show: List[np.ndarray] = []
    titles_2_show: List[str] = []

    for _ in range(10):
        r: int = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append(f'training image [{r}] = {y_train[r]}')

    for _ in range(5):
        r: int = random.randint(1, 10000)
        images_2_show.append(x_test[r])
        titles_2_show.append(f'test image [{r}] = {y_test[r]}')

    show_images(images_2_show, titles_2_show)


if __name__ == "__main__":
    main()