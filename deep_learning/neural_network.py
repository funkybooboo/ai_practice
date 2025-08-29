import random
from typing import List, Callable

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
        self.layers: List[List[Neuron]] = []
        self.input_size: int = input_size
        self.batch_size = batch_size  # Store batch size

        sizes: List[int] = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(sizes)):
            last_size = sizes[i - 1]
            layer = [Neuron(last_size, activation, activation_derivative, lr) for _ in range(sizes[i])]
            self.layers.append(layer)

    def _forward_propagate(self, xs: List[float]) -> List[float]:
        """Run one input through the network"""
        for layer in self.layers:
            xs = [neuron.forward(xs) for neuron in layer]
        return xs

    def _backward_propagate(self, deltas: List[float]) -> None:
        """Run backpropagation step and update weights"""
        for layer in reversed(self.layers):
            new_deltas = []
            for j, neuron in enumerate(layer):
                grads = neuron.backward(deltas[j])   # grads: contribution to previous layer
                if not new_deltas:
                    new_deltas = [0.0] * len(grads)
                for k, g in enumerate(grads):
                    new_deltas[k] += g
            deltas = new_deltas   # pass error backward

    def predict(self, xss: List[List[float]]) -> List[int]:
        """Predict class labels for a batch of inputs"""
        results = []
        for xs in xss:
            outputs = self._forward_propagate(xs)
            pred = outputs.index(max(outputs))  # argmax
            results.append(pred)
        return results

    def fit(self, xss: List[List[float]], ys: List[int], epochs: int):
        """Train the network using backpropagation"""
        for epoch in range(epochs):
            correct = 0

            # Shuffle data at the start of each epoch
            data = list(zip(xss, ys))
            random.shuffle(data)
            xss_shuffled, ys_shuffled = zip(*data)

            # Mini-batch training
            for start_idx in range(0, len(xss), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(xss))
                batch_xs = xss_shuffled[start_idx:end_idx]
                batch_ys = ys_shuffled[start_idx:end_idx]

                # Train on mini-batch
                for xs, y in zip(batch_xs, batch_ys):
                    outputs = self._forward_propagate(xs)
                    pred = outputs.index(max(outputs))
                    if pred == y:
                        correct += 1

                    target = [0.0] * len(outputs)
                    target[y] = 1.0
                    deltas = [t - o for t, o in zip(target, outputs)]
                    self._backward_propagate(deltas)

            # Epoch accuracy
            accuracy = (correct / len(ys)) * 100
            print(f"Epoch {epoch + 1}/{epochs} Accuracy: {accuracy:.2f}%")
