import random
from typing import Callable, List


class Neuron:
    def __init__(self, input_size: int, activation: Callable[[float], float], activation_derivative: Callable[[float], float], lr: float):
        self.ws = [random.uniform(-0.5,0.5) for _ in range(input_size)]
        self.b = random.uniform(-0.5,0.5)
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.lr = lr
        self.last_input = []
        self.last_output = 0

    def forward(self, xs: List[float]) -> float:
        self.last_input = xs
        s = sum(x*w for x,w in zip(xs,self.ws)) + self.b
        self.last_output = self.activation(s)
        return self.last_output

    def backward(self, delta: float):
        # delta = error * f'(s)
        deriv = self.activation_derivative(self.last_output)
        for i in range(len(self.ws)):
            self.ws[i] += self.lr * delta * deriv * self.last_input[i]
        self.b += self.lr * delta * deriv
        return [delta * deriv * w for w in self.ws]  # gradient for previous layer
