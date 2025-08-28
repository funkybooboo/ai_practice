import random
from typing import List

from deep_learning.activation_functions import sigmoid_function
from deep_learning.list_utils import softmax_list, argmax_2d
from deep_learning.perceptron import Perceptron


class Mlp:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, lr: float = 0.1) -> None:
        """
        Build an MLP with any number of hidden layers.
        :param input_size: number of input features
        :param hidden_sizes: list of hidden layer sizes
        :param output_size: number of output neurons/classes
        :param lr: learning rate
        """
        random.seed(42)
        self.lr = lr

        # Build layers as lists of Perceptrons
        self.layers: List[List[Perceptron]] = []

        prev_size = input_size
        for h_size in hidden_sizes:
            self.layers.append([Perceptron(prev_size, sigmoid_function, lr) for _ in range(h_size)])
            prev_size = h_size

        # Output layer
        self.layers.append([Perceptron(prev_size, lambda z: z, lr) for _ in range(output_size)])  # linear output

    @staticmethod
    def _one_hot(y: List[float], num_classes: int) -> List[List[float]]:
        return [[1.0 if i == label else 0.0 for i in range(num_classes)] for label in y]

    def _forward(self, X: List[List[float]]) -> List[List[float]]:
        """Forward pass through all layers."""
        activations = X
        for layer in self.layers[:-1]:
            activations = [[neuron.predict(x) for neuron in layer] for x in activations]
        # Output layer
        output_layer = self.layers[-1]
        out = [[neuron.predict(x) for neuron in output_layer] for x in activations]
        return softmax_list(out)

    def predict(self, xss: List[List[float]]) -> List[float]:
        return argmax_2d(self._forward(xss))

    def fit(self, xss: List[List[float]], ys: List[float], epochs: int = 5) -> None:
        y_onehot = self._one_hot(ys, len(self.layers[-1]))

        for epoch in range(epochs):
            # Forward pass up to last hidden layer
            activations = xss
            for layer in self.layers[:-1]:
                activations = [[neuron.predict(x) for neuron in layer] for x in activations]

            # Train output layer independently using Perceptron learning
            for j, neuron in enumerate(self.layers[-1]):
                target_j = [row[j] for row in y_onehot]
                neuron.train(activations, target_j, epochs=1)

            preds = self.predict(xss)
            acc = sum(p == t for p, t in zip(preds, ys)) / len(ys)
            print(f"Epoch {epoch + 1}/{epochs} - Acc: {acc:.2%}")
