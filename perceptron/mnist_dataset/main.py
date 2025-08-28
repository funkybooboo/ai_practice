from os.path import join

from perceptron.mnist_dataset.mnist_dataloader import MnistDataloader
from perceptron.perceptron import perceptron


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
    (training_data, test_data) = mnist_dataloader.load_data()

    print(training_data)

    


if __name__ == "__main__":
    main()
