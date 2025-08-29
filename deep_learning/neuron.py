import random
from typing import Callable, List


class Neuron:
    def __init__(self, input_size: int, activation: Callable[[float], float],
                 activation_derivative: Callable[[float], float], lr: float):
        # Xavier Initialization for weights
        self.ws = [random.uniform(-1 / (input_size ** 0.5), 1 / (input_size ** 0.5)) for _ in range(input_size)]
        self.b = 0  # Bias initialized to 0 (can experiment with small random values)
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.lr = lr
        self.last_input = []
        self.last_output = 0

    def forward(self, xs: List[float]) -> float:
        """Calculate output for a given input."""
        self.last_input = xs
        s = sum(x * w for x, w in zip(xs, self.ws)) + self.b
        self.last_output = self.activation(s)
        return self.last_output

    def backward(self, delta: float):
        """Update weights and bias using backpropagation and return gradient for previous layer."""
        # Compute the derivative of the activation function
        deriv = self.activation_derivative(self.last_output)

        # Update weights and bias
        for i in range(len(self.ws)):
            self.ws[i] += self.lr * delta * deriv * self.last_input[i]
        self.b += self.lr * delta * deriv

        # Return gradient for the previous layer
        return [delta * deriv * w for w in self.ws]
