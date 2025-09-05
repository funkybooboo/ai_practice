# MNIST Neural Network Project

This project implements a simple fully connected neural network (MLP) from scratch to train and evaluate on the **MNIST handwritten digits dataset**. It includes utilities for loading the MNIST dataset, flattening and normalizing images, training a network, saving/loading models, and visualizing predictions.

---

## Table of Contents

* [Setup](#setup)
* [Project Structure](#project-structure)
* [Training the Model](#training-the-model)
* [Using a Saved Model](#using-a-saved-model)
* [Visualizing MNIST Images](#visualizing-mnist-images)
* [Configuration](#configuration)

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/funkybooboo/ai_practice
cd deep_learning/mnist_dataset
```

2. **Install dependencies**
   This project uses Python 3.9+ and requires the following packages:

```bash
pip install matplotlib
```

3. **Download the MNIST dataset**
   You need the MNIST dataset in IDX format. You can download it from [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) or [Yann LeCun's site](http://yann.lecun.com/exdb/mnist/).
   Place the dataset files under the `./archive` folder with the following structure:

```
archive/
├─ train-images-idx3-ubyte
├─ train-labels-idx1-ubyte
├─ t10k-images-idx3-ubyte
├─ t10k-labels-idx1-ubyte
```

---

## Project Structure

```
mnist_dataset/
│
├─ archive/                     # Folder containing MNIST dataset files
├─ mnist_dataloader.py          # Loads and preprocesses MNIST images
├─ create_model.py              # Script to train a new neural network model
├─ use_model.py                 # Script to load a saved model and evaluate/test it
├─ model<timestamp>.pkl         # Example saved model
```

* **`mnist_dataloader.py`**: Handles reading MNIST IDX files and normalizes the images.
* **`create_model.py`**: Trains a neural network on MNIST and saves the trained model.
* **`use_model.py`**: Loads a saved model and evaluates its accuracy on the test set, with visualization.

---

## Training the Model

To train a new neural network:

1. Adjust configuration at the top of `create_model.py`:

```python
TRAIN_SIZE = 10000        # Number of training samples to use
TEST_SIZE = 2000          # Number of test samples to use
NN_HIDDEN_SIZES = [16,16] # Hidden layer sizes
BATCH_SIZE = 50
LR = 0.001
EPOCHS = 10
```

2. Run the training script:

```bash
python create_model.py
```

This will:

* Load and preprocess MNIST data
* Train a fully connected network
* Print test accuracy
* Save the trained model in the format `model<timestamp>.pkl`

---

## Using a Saved Model

To evaluate a saved model:

1. Update `MODEL_PATH` in `use_model.py` with your saved model file.

```python
MODEL_PATH = './model17565163193815045.pkl'
```

2. Run the script:

```bash
python use_model.py
```

This will:

* Load the MNIST test set
* Load your saved model
* Compute and print test accuracy
* Display random test images with true and predicted labels

---

## Visualizing MNIST Images

You can use `mnist_dataloader.py` directly to visualize random images:

```bash
python mnist_dataloader.py
```

This script will randomly select and display some training and test images with their labels.

---

This setup allows you to train a simple neural network on MNIST, save and reuse models, and visualize results.
