from typing import Callable, Optional
import numpy as np


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Callable[[np.ndarray], np.ndarray],
        activation_derivative: Callable[[np.ndarray], np.ndarray],
        learning_rate: float
    ) -> None:
        # Initialize weights and biases
        self.W: np.ndarray = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        self.b: np.ndarray = np.zeros((output_size, 1))

        # Store activation functions
        self.activation: Callable[[np.ndarray], np.ndarray] = activation
        self.activation_derivative: Callable[[np.ndarray], np.ndarray] = activation_derivative
        self.learning_rate: float = learning_rate

        # Cache for forward/backward pass
        self.Z: Optional[np.ndarray] = None
        self.A: Optional[np.ndarray] = None
        self.X: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        # Forward pass: compute output of layer
        self.X = X  # Store input
        self.Z = X @ self.W.T + self.b.T  # Linear step
        self.A = self.activation(self.Z)  # Apply activation
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        # Backward pass: compute gradient w.r.t. input and update weights
        dZ: np.ndarray = dA * self.activation_derivative(self.A)  # Gradient of linear output
        dW: np.ndarray = dZ.T @ self.X / self.X.shape[0]          # Gradient w.r.t. weights
        db: np.ndarray = np.mean(dZ, axis=0, keepdims=True).T     # Gradient w.r.t. biases

        dX: np.ndarray = dZ @ self.W                              # Gradient w.r.t. input

        # Update parameters
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

        return dX
