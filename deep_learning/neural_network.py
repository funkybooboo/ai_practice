import math
import random
from typing import List, Callable, Tuple, Generator
import pickle

from deep_learning.neuron import Neuron


class NeuralNetwork:
    def __init__(self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: Callable[[float], float],
        activation_derivative: Callable[[float], float],
        learning_rate: float,
        batch_size: int,
    ):
        self.features_size: int = input_size
        self.batch_size = batch_size

        self.layers: List[List[Neuron]] = []
        sizes: List[int] = [input_size] + hidden_sizes + [output_size]
        for i in range(1, len(sizes)):
            last_size = sizes[i - 1]
            is_output_neuron = (i == len(sizes) - 1)
            layer: List[Neuron] = [Neuron(last_size, activation, activation_derivative, learning_rate, is_output_neuron=is_output_neuron)
                     for _ in range(sizes[i])]
            self.layers.append(layer)

    def predict(self, features_table: List[List[float]]) -> List[int]:
        """Predict class labels for a batch of inputs"""
        predictions = []
        for features in features_table:
            outputs = self._forward_propagate(features)
            prediction = outputs.index(max(outputs))  # argmax
            predictions.append(prediction)
        return predictions

    def fit(self, features_table: List[List[float]], labels: List[int], epochs: int):
        for epoch in range(epochs):
            features_table, labels = self._shuffle_data(features_table, labels)

            total_loss: float = 0
            processed_count: int = 0

            for features_batch, labels_batch in self._mini_batches(features_table, labels):
                for features, label in zip(features_batch, labels_batch):
                    outputs = self._forward_propagate(features)

                    loss, error_deltas = self._compute_loss_and_error_deltas(outputs, label)

                    total_loss += loss
                    processed_count += 1

                    self._backward_propagate(error_deltas)

            self._log_epoch_stats(epoch, epochs, total_loss, processed_count, features_table, labels)

    def _forward_propagate(self, features: List[float]) -> List[float]:
        for i, layer in enumerate(self.layers):
            features: List[float] = [neuron.forward(features) for neuron in layer]
        return features

    def _backward_propagate(self, error_deltas: List[float]) -> None:
        """Run backpropagation step and update weights"""
        for layer in reversed(self.layers):
            next_error_deltas = []
            for j, neuron in enumerate(layer):
                gradients = neuron.backward(error_deltas[j])  # gradients: contribution to previous layer
                if not next_error_deltas:
                    next_error_deltas = [0.0] * len(gradients)
                for k, gradiant in enumerate(gradients):
                    next_error_deltas[k] += gradiant
            error_deltas = next_error_deltas

    @staticmethod
    def _shuffle_data(features_table: List[List[float]], labels: List[int]) -> Tuple[List[List[float]], List[int]]:
        """Shuffle the features and labels together"""
        dataset: List[Tuple[List[float], int]] = list(zip(features_table, labels))
        random.shuffle(dataset)
        randomized_features_table, randomized_labels = zip(*dataset) # the features and labels indexes still match
        return randomized_features_table, randomized_labels

    def _mini_batches(self, features_table: List[List[float]], labels: List[int]) -> Generator[tuple[list[list[float]], list[int]], None, None]:
        """Generate mini-batches from the dataset"""
        for start_index in range(0, len(features_table), self.batch_size):
            end_index = min(start_index + self.batch_size, len(features_table))
            yield features_table[start_index:end_index], labels[start_index:end_index]

    @staticmethod
    def _compute_loss_and_error_deltas(outputs: List[float], label: int) -> Tuple[float, List[float]]:
        """Compute loss and error_deltas for backpropagation"""
        targets = [0.0] * len(outputs)
        targets[label] = 1.0

        loss = 0
        error_deltas = []
        for output, target in zip(outputs, targets):
            error_delta = output - target
            loss += math.pow(error_delta, 2)
            error_deltas.append(error_delta)

        return loss, error_deltas

    def _log_epoch_stats(self, epoch: int, epochs: int, total_loss: float, processed_count: int,
                         features_table: List[List[float]], labels: List[int]):
        """Log the epoch stats including loss and accuracy"""
        print(f"Epoch {epoch + 1}/{epochs}")
        average_loss = total_loss / processed_count
        print(f"\tAverage Loss: {average_loss:.2f}")

        predictions = self.predict(features_table)
        accuracy = sum(p == y for p, y in zip(predictions, labels)) / len(labels) * 100
        print(f"\tAccuracy: {accuracy:.2f}%")

    def save(self, filename: str) -> None:
        """Save the neural network to a file using pickle"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f) # type: ignore
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename: str) -> 'NeuralNetwork':
        """Load a neural network from a file using pickle"""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
