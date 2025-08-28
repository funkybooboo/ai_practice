from perceptron.mnist_dataset.mnist_dataloader import MnistDataloader
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import random

from perceptron.multi_layer_perceptron import MultiLayerPerceptron


def main():
    input_path = './archive'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    mnist_dataloader = MnistDataloader(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath
    )
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    # Flatten + normalize images
    X_train = np.array([img.flatten() for img in x_train], dtype=np.float32) / 255.0
    y_train = np.array(y_train)
    X_test = np.array([img.flatten() for img in x_test], dtype=np.float32) / 255.0
    y_test = np.array(y_test)

    # Build MLP: 784 -> 20 -> 20 -> 10
    mlp = MultiLayerPerceptron(input_size=784, hidden_size=20, output_size=10)

    # Train
    mlp.fit(X_train, y_train, lr=0.1, epochs=5, batch_size=128)

    # Evaluate
    preds = mlp.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"Test accuracy: {acc:.2%}")

    # Pick some random test images
    num_examples = 10
    indices = random.sample(range(len(X_test)), num_examples)

    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(indices):
        img = X_test[idx].reshape(28, 28)  # reshape back to 28x28
        true_label = y_test[idx]
        pred_label = preds[idx]

        plt.subplot(2, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
