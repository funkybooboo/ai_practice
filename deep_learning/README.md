# Understanding the `NeuralNetwork` and `Neuron` Classes

This document explains how the `NeuralNetwork` and `Neuron` classes function within a simple machine learning framework. These classes are used to build, train, and use a multi-layer perceptron (MLP) model, which is a type of neural network.

### **Overview of Neural Networks**

A neural network is made up of layers of connected neurons. Each neuron performs a mathematical operation on input data and passes the result to the next layer. The neural network learns from data by adjusting the connections (weights) between neurons based on the errors it makes in its predictions. This is done through a process called backpropagation.

### **The `NeuralNetwork` Class**

The `NeuralNetwork` class is the high-level structure that manages the entire neural network. It defines how the network is built, how it trains, and how it makes predictions.

#### **How It Works:**

1. **Building the Network:**

   * The network is made up of layers. Each layer contains neurons, and each neuron has connections to the neurons in the previous layer.
   * The `NeuralNetwork` class allows you to define how many layers your network has and how many neurons are in each layer.
   * The network also requires an activation function, which is a mathematical function that determines whether a neuron should "fire" or not based on its input. Common examples include the ReLU (Rectified Linear Unit) function.

2. **Training the Network (Learning):**

   * To train the network, you provide it with input data (features) and the correct output (labels).
   * The network makes predictions, checks how wrong it is (calculates the loss), and adjusts its internal parameters (weights) using a method called backpropagation.
   * During training, the network updates its weights to minimize the difference between its predictions and the true output.

3. **Making Predictions:**

   * After training, the network can be used to make predictions on new data.
   * The network processes input data through each of its layers and returns an output, which can be used for classification or regression tasks.

4. **Saving and Loading the Network:**

   * Once the network is trained, you can save it to a file, so you don’t need to retrain it every time. You can later load the saved model to make predictions.

---

### **The `Neuron` Class**

Each neuron in a neural network is responsible for processing its inputs and passing the result to the next layer. The `Neuron` class represents a single neuron in the network.

#### **How It Works:**

1. **Input Processing:**

   * Each neuron receives a set of inputs (features) from the previous layer. These inputs are weighted, meaning each input has a "strength" that determines how much it influences the neuron’s output.
   * The neuron calculates a weighted sum of its inputs, adds a bias term (which allows the neuron to shift its output), and passes this sum through an activation function.

2. **Activation Function:**

   * The activation function determines whether a neuron "fires" or not. Common activation functions include ReLU (which outputs the input if it's positive, and zero otherwise) or sigmoid (which squashes the output between 0 and 1).
   * The purpose of the activation function is to introduce non-linearity into the network, allowing it to model complex patterns.

3. **Learning from Errors (Backpropagation):**

   * During training, each neuron helps the network learn by updating its weights. When the network makes an incorrect prediction, the neuron adjusts its weights to reduce the error.
   * The `Neuron` class uses backpropagation to compute how much each weight contributed to the error, then adjusts the weights by a small amount (determined by the learning rate).

4. **Weight Update:**

   * Each time the neuron processes an input and contributes to an error, it updates its weights by subtracting the calculated gradient (scaled by the learning rate) from each weight.
   * This process is how the neuron "learns" to make better predictions over time.

---

### **Key Concepts in Action**

* **Forward Propagation:** When you feed input data into the network, it flows from the input layer, through hidden layers, and finally to the output layer. This is called "forward propagation." Each neuron applies its weights and activation function to produce an output that is passed to the next neuron.

* **Backward Propagation:** After the network makes a prediction, it compares the result to the correct label. The difference between the predicted output and the true label is the error. Backpropagation is the process by which this error is sent back through the network, and the weights are adjusted to reduce the error in future predictions.

* **Mini-Batch Training:** The training process is typically done in mini-batches, where a small subset of the training data is used in each iteration. This helps to speed up the learning process and stabilize the gradient updates.

* **Loss Function:** The loss function measures how far off the network's predictions are from the true labels. Common loss functions include Mean Squared Error (for regression) or Cross-Entropy Loss (for classification). The network's goal is to minimize this loss during training.

---

### **In Summary:**

* The **`NeuralNetwork` class** organizes the overall network and handles training and prediction processes, including managing layers of neurons, forward and backward propagation, and loss calculation.
* The **`Neuron` class** represents a single computational unit that processes inputs, applies weights, uses an activation function, and adjusts its weights based on errors during training.

Together, these classes allow you to build, train, and evaluate a multi-layer neural network that can be used for a variety of machine learning tasks, such as image classification, regression, and more.
