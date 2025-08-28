import numpy as np
from typing import Tuple

from perceptron.activation_functions import sigmoid_function, softmax_function
from perceptron.encoders import one_hot
from perceptron.loss_functions import cross_entropy


class MultiLayerPerceptron:
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        rng = np.random.default_rng(seed=42)
        # Layer 1
        self.W1 = rng.normal(0, 0.1, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        # Layer 2
        self.W2 = rng.normal(0, 0.1, (hidden_size, hidden_size))
        self.b2 = np.zeros((1, hidden_size))
        # Output layer
        self.W3 = rng.normal(0, 0.1, (hidden_size, output_size))
        self.b3 = np.zeros((1, output_size))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, dict]:
        z1 = X @ self.W1 + self.b1
        a1 = sigmoid_function(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid_function(z2)
        z3 = a2 @ self.W3 + self.b3
        a3 = softmax_function(z3)

        cache = {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3}
        return a3, cache

    def backward(self, y_true: np.ndarray, cache: dict, lr: float) -> None:
        m = y_true.shape[0]
        a3 = cache["a3"]

        dz3 = a3 - y_true
        dW3 = cache["a2"].T @ dz3 / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = (dz3 @ self.W3.T) * (cache["a2"] * (1 - cache["a2"]))
        dW2 = cache["a1"].T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = (dz2 @ self.W2.T) * (cache["a1"] * (1 - cache["a1"]))
        dW1 = cache["X"].T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def predict(self, X: np.ndarray) -> np.ndarray:
        a3, _ = self.forward(X)
        return np.argmax(a3, axis=1)

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 10, batch_size: int = 64) -> None:
        y_onehot = one_hot(y, 10)
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y_onehot[i:i+batch_size]

                y_pred, cache = self.forward(X_batch)
                self.backward(y_batch, cache, lr)

            y_pred, _ = self.forward(X)
            loss = cross_entropy(y_pred, y_onehot)
            acc = np.mean(np.argmax(y_pred, axis=1) == y)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f} - Acc: {acc:.4f}")
