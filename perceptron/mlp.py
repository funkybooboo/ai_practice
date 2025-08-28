import random
from typing import List, Callable

from perceptron.activation_functions import sigmoid_function
from perceptron.list_utils import softmax_list, argmax_2d
from perceptron.perceptron import perceptron, train_perceptron


class Mlp:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        random.seed(42)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases for each layer
        self.W1 = [[random.gauss(0, 0.1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [0.0 for _ in range(hidden_size)]

        self.W2 = [[random.gauss(0, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.b2 = [0.0 for _ in range(hidden_size)]

        self.W3 = [[random.gauss(0, 0.1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b3 = [0.0 for _ in range(output_size)]

    @staticmethod
    def _one_hot(y: List[int], num_classes: int) -> List[List[float]]:
        """
        Convert integer class labels to one-hot encoding using lists.

        Parameters:
        y: list of class labels (ints)
        num_classes: number of classes

        Returns:
        List[List[float]]: one-hot encoded labels
        """
        one_hot_labels: List[List[float]] = []
        for label in y:
            vec = [0.0] * num_classes
            vec[label] = 1.0
            one_hot_labels.append(vec)
        return one_hot_labels

    @staticmethod
    def _forward_layer(X: List[List[float]], W: List[List[float]], b: List[float],
                       a: Callable[[float], float]) -> List[List[float]]:
        return [[perceptron(x, [W[i][j] for i in range(len(W))], b[j], a) for j in range(len(b))] for x in X]

    def _forward(self, X: List[List[float]]) -> List[List[float]]:
        h1 = [[perceptron(x, [self.W1[i][j] for i in range(len(self.W1))], self.b1[j], sigmoid_function) for j in
               range(self.hidden_size)] for x in X]
        h2 = [
            [perceptron(h1_row, [self.W2[i][j] for i in range(len(self.W2))], self.b2[j], sigmoid_function) for j in
             range(self.hidden_size)] for h1_row in h1]
        out = [[perceptron(h2_row, [self.W3[i][j] for i in range(len(self.W3))], self.b3[j], lambda z: z) for j in
                range(self.output_size)] for h2_row in h2]
        return softmax_list(out)

    def predict(self, X: List[List[float]]) -> List[int]:
        return argmax_2d(self._forward(X))

    def fit(self, X: List[List[float]], y: List[int], lr: float = 0.1, epochs: int = 5) -> None:
        # Convert y to one-hot
        y_onehot = self._one_hot(y, self.output_size)
        for epoch in range(epochs):
            # Train each neuron in each layer independently
            # Layer 1
            for j in range(self.hidden_size):
                ys_j = [sigmoid_function(y_val[j]) for y_val in y_onehot]  # target for hidden layer? approximation
                self.W1[j], self.b1[j] = train_perceptron(X, ys_j, [self.W1[i][j] for i in range(self.input_size)],
                                                          self.b1[j], sigmoid_function, lr, 1)
            # Could do similarly for Layer2 and output layer
            preds = self.predict(X)
            acc = sum(p == t for p, t in zip(preds, y)) / len(y)
            print(f"Epoch {epoch + 1}/{epochs} - Acc: {acc:.2%}")
