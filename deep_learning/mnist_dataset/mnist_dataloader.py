from __future__ import annotations
import struct
import random
import matplotlib.pyplot as plt
from os.path import join
from typing import List, Tuple


class MnistDataloader:
    def __init__(
        self,
        training_images_filepath: str,
        training_labels_filepath: str,
        test_images_filepath: str,
        test_labels_filepath: str
    ) -> None:
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    @staticmethod
    def read_images_labels(
        images_filepath: str,
        labels_filepath: str
    ) -> Tuple[List[List[List[float]]], List[int]]:
        # Read labels
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Magic number mismatch, expected 2049, got {magic}')
            labels = list(file.read())

        # Read images
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Magic number mismatch, expected 2051, got {magic}')
            image_data = list(file.read())

        images: List[List[List[float]]] = []
        for i in range(size):
            img_flat = image_data[i*rows*cols:(i+1)*rows*cols]
            img = []
            for r in range(rows):
                row = [float(img_flat[r*cols + c]) / 255.0 for c in range(cols)]
                img.append(row)
            images.append(img)

        return images, labels

    def load_data(
        self
    ) -> Tuple[Tuple[List[List[List[float]]], List[int]], Tuple[List[List[List[float]]], List[float]]]:
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


def show_images(images: List[List[List[float]]], title_texts: List[str]) -> None:
    cols = 5
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, 8))
    for idx, (img, title) in enumerate(zip(images, title_texts)):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.show()


def main():
    input_path = './archive'
    mnist_dataloader = MnistDataloader(
        join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'),
        join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'),
        join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'),
        join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    )

    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Show some random images
    images_to_show = []
    titles_to_show = []
    for _ in range(10):
        idx = random.randint(0, len(x_train)-1)
        images_to_show.append(x_train[idx])
        titles_to_show.append(f"train[{idx}] = {y_train[idx]}")
    for _ in range(5):
        idx = random.randint(0, len(x_test)-1)
        images_to_show.append(x_test[idx])
        titles_to_show.append(f"test[{idx}] = {y_test[idx]}")

    show_images(images_to_show, titles_to_show)


if __name__ == "__main__":
    main()
