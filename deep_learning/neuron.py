import math
import random
from typing import Callable, List


class Neuron:
    def __init__(self, input_size: int, activation: Callable[[float], float],
                 activation_derivative: Callable[[float], float], lr: float, is_output=False):
        # Xavier Initialization for weights
        self.ws = [random.uniform(-1, 1) * math.sqrt(1 / input_size)
                   for _ in range(input_size)]
        self.b = random.uniform(-0.01, 0.01)
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.lr = lr
        self.last_input = []
        self.last_output = 0
        self.is_output = is_output

    def forward(self, xs: List[float]) -> float:
        """Calculate output for a given input."""
        self.last_input = xs
        s = sum(x * w for x, w in zip(xs, self.ws)) + self.b
        self.last_output = self.activation(s)
        return self.last_output

    def backward(self, delta: float):
        if self.is_output:
            grad = delta   # delta = y_pred - y_true
        else:
            grad = delta * self.activation_derivative(self.last_output)

        for i in range(len(self.ws)):
            self.ws[i] -= self.lr * grad * self.last_input[i]
        self.b -= self.lr * grad

        return [grad * w for w in self.ws]
