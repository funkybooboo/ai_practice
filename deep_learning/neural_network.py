import math
import random
from typing import List, Callable
import pickle

from deep_learning.neuron import Neuron


class NeuralNetwork:
    def __init__(self,
         input_size: int,
         hidden_sizes: List[int],
         output_size: int,
         activation: Callable[[float], float],
         activation_derivative: Callable[[float], float],
         lr: float,
         batch_size: int,
     ):
        self.input_size: int = input_size
        self.batch_size = batch_size  # Store batch size

        self.layers: List[List[Neuron]] = []
        sizes: List[int] = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(sizes)):
            last_size = sizes[i - 1]
            is_output = (i == len(sizes) - 1)
            layer = [Neuron(last_size, activation, activation_derivative, lr, is_output=is_output)
                     for _ in range(sizes[i])]
            self.layers.append(layer)

    def _forward_propagate(self, xs: List[float]) -> List[float]:
        for i, layer in enumerate(self.layers):
            xs = [neuron.forward(xs) for neuron in layer]
        return xs

    def _backward_propagate(self, deltas: List[float]) -> None:
        """Run backpropagation step and update weights"""
        for layer in reversed(self.layers):
            new_deltas = []
            for j, neuron in enumerate(layer):
                grads = neuron.backward(deltas[j])  # grads: contribution to previous layer
                if not new_deltas:
                    new_deltas = [0.0] * len(grads)
                for k, g in enumerate(grads):
                    new_deltas[k] += g
            deltas = new_deltas

    def predict(self, xss: List[List[float]]) -> List[int]:
        """Predict class labels for a batch of inputs"""
        results = []
        for xs in xss:
            outputs = self._forward_propagate(xs)
            pred = outputs.index(max(outputs))  # argmax
            results.append(pred)
        return results

    def fit(self, xss: List[List[float]], ys: List[int], epochs: int):
        for epoch in range(epochs):
            # shuffle data
            data = list(zip(xss, ys))
            random.shuffle(data)
            xss_shuffled, ys_shuffled = zip(*data)

            total_cost: float = 0
            cost_count: int = 0

            # train in mini-batches
            for start_idx in range(0, len(xss), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(xss))
                batch_xs = xss_shuffled[start_idx:end_idx]
                batch_ys = ys_shuffled[start_idx:end_idx]

                for xs, y in zip(batch_xs, batch_ys):
                    outputs = self._forward_propagate(xs)

                    target = [0.0] * len(outputs)
                    target[y] = 1.0

                    cost: float = 0
                    for o, t in zip(outputs, target):
                        cost += math.pow(o - t, 2)
                    total_cost += cost
                    cost_count += 1

                    deltas = [o - t for o, t in zip(outputs, target)]
                    self._backward_propagate(deltas)

            print(f"Epoch {epoch + 1}/{epochs}")

            average_cost: float = total_cost / cost_count
            print(f"\tAverage Cost: {average_cost:.2f}")

            preds = self.predict(xss)
            accuracy = sum(p == y for p, y in zip(preds, ys)) / len(ys) * 100
            print(f"\tAccuracy: {accuracy:.2f}%")

    def save(self, filename: str) -> None:
        """Save the neural network to a file using pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filename}")

    @staticmethod
    def load_model(filename: str) -> 'NeuralNetwork':
        """Load a neural network from a file using pickle"""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
