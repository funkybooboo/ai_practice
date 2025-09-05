import pickle
from typing import List, Callable, Tuple
import numpy as np

from basics.supervised_learning.classification.with_numpy.layer import Layer


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: Callable[[np.ndarray], np.ndarray],
        activation_derivative: Callable[[np.ndarray], np.ndarray],
        learning_rate: float,
        batch_size: int
    ) -> None:
        # Initialize layers
        self.layers: List[Layer] = []
        sizes: List[int] = [input_size] + hidden_sizes + [output_size]
        for i in range(len(sizes) - 1):
            self.layers.append(
                Layer(
                    sizes[i],
                    sizes[i + 1],
                    activation,
                    activation_derivative,
                    learning_rate
                )
            )
        self.batch_size: int = batch_size

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Forward pass and return class predictions
        outputs: np.ndarray = self._forward_propagate(X)
        return np.argmax(outputs, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int) -> None:
        # Train network for given number of epochs
        for epoch in range(epochs):
            idx: np.ndarray = np.random.permutation(len(X))  # Shuffle data
            X, y = X[idx], y[idx]

            total_loss: float = 0.0
            for start in range(0, len(X), self.batch_size):
                end: int = start + self.batch_size
                X_batch, y_batch = X[start:end], y[start:end]

                outputs: np.ndarray = self._forward_propagate(X_batch)
                loss, dA = self._compute_loss_and_gradients(outputs, y_batch)
                total_loss += loss * len(X_batch)

                self._backward_propagate(dA)

            avg_loss: float = total_loss / len(X)
            acc: float = np.mean(self.predict(X) == y) * 100
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.2f} - Acc: {acc:.2f}%")

    def _forward_propagate(self, X: np.ndarray) -> np.ndarray:
        # Run data through layers
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def _backward_propagate(self, dA: np.ndarray) -> None:
        # Backpropagate gradients through layers
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    @staticmethod
    def _compute_loss_and_gradients(outputs: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        # Compute MSE loss and gradient
        batch_size, num_classes = outputs.shape
        targets: np.ndarray = np.zeros((batch_size, num_classes))
        targets[np.arange(batch_size), y] = 1.0
        loss: float = float(np.mean(np.square(outputs - targets)))
        dA: np.ndarray = 2 * (outputs - targets) / batch_size
        return loss, dA

    def save(self, filename: str) -> None:
        # Save model with pickle
        with open(filename, 'wb') as f:
            pickle.dump(self, f)  # type: ignore
        print(f"Model saved to {filename}")

    @staticmethod
    def load(filename: str) -> 'NeuralNetwork':
        # Load model with pickle
        with open(filename, 'rb') as f:
            model: NeuralNetwork = pickle.load(f)
        print(f"Model loaded from {filename}")
        return model
