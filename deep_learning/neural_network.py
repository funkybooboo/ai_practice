from typing import List, Optional
from deep_learning.activation_functions import sigmoid_function, sigmoid_derivative
from deep_learning.list_utils import argmax_2d
from deep_learning.neuron import Neuron


class NeuralNetwork:
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, lr: float = 0.1) -> None:
        """
        Build a feedforward neural network with any number of hidden layers.
        :param input_size: number of input features
        :param hidden_sizes: list of hidden layer sizes
        :param output_size: number of output neurons/classes
        :param lr: learning rate
        """
        self.lr = lr

        # Build layers
        self.layers: List[List[Neuron]] = []
        prev_size = input_size
        for h_size in hidden_sizes:
            self.layers.append([Neuron(prev_size, sigmoid_function, lr) for _ in range(h_size)])
            prev_size = h_size

        # Output layer (linear for softmax)
        self.layers.append([Neuron(prev_size, lambda z: z, lr) for _ in range(output_size)])

    @staticmethod
    def _one_hot(y: List[int], num_classes: int) -> List[List[float]]:
        return [[1.0 if i == label else 0.0 for i in range(num_classes)] for label in y]

    def _forward(self, x: List[float]) -> List[List[float]]:
        """
        Forward pass for a single input sample.
        Returns activations of all layers including output.
        """
        activations = [x]
        for layer in self.layers:
            x = [neuron.predict(activations[-1]) for neuron in layer]
            activations.append(x)
        return activations

    def _backpropagate(self, activations: List[List[float]], y_true: List[float]) -> None:
        deltas: List[Optional[List[float]]] = [None] * len(self.layers)

        # Output layer delta
        output_activation = activations[-1]
        deltas[-1] = [t - o for t, o in zip(y_true, output_activation)]

        # Hidden layers
        for l in reversed(range(len(self.layers) - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            deltas_l = []
            for i, neuron in enumerate(layer):
                delta_sum = sum(deltas[l + 1][j] * next_neuron.ws[i] for j, next_neuron in enumerate(next_layer))
                delta_i = delta_sum * sigmoid_derivative(activations[l + 1][i])
                deltas_l.append(delta_i)
            deltas[l] = deltas_l

        # Update weights
        for l, layer in enumerate(self.layers):
            inputs = activations[l]
            for j, neuron in enumerate(layer):
                neuron.fit([inputs], [deltas[l][j]], epochs=1)

    def predict(self, xss: List[List[float]]) -> List[int]:
        """
        Predict labels for multiple inputs.
        """
        all_outputs = []
        for x in xss:
            out = self._forward(x)[-1]
            all_outputs.append(out)
        return argmax_2d(all_outputs)

    def fit(self, xss: List[List[float]], ys: List[int], epochs: int) -> None:
        """
        Train the neural network using full backpropagation.
        """
        y_onehot = self._one_hot(ys, len(self.layers[-1]))

        for epoch in range(epochs):
            for x, y_true in zip(xss, y_onehot):
                activations = self._forward(x)
                self._backpropagate(activations, y_true)

            preds = self.predict(xss)
            acc = sum(p == t for p, t in zip(preds, ys)) / len(ys)
            print(f"Epoch {epoch+1}/{epochs} - Acc: {acc:.2%}")
