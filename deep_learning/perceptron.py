from collections.abc import Callable
from typing import List
import random


class Perceptron:
    def __init__(self, input_size: int, activation: Callable[[float], float], lr: float = 0.01) -> None:
        random.seed(42)
        self.ws: List[float] = [random.gauss(0, 0.1) for _ in range(input_size)]
        self.b: float = 0.0
        self.activation: Callable[[float], float] = activation
        self.lr: float = lr

    def predict(self, xs: List[float]) -> float:
        """Compute the output of the deep_learning for given inputs."""
        s = sum(x * w for x, w in zip(xs, self.ws)) + self.b
        return self.activation(s)

    def train(self, xss: List[List[float]], ys: List[float], epochs: int = 1) -> None:
        """Train the deep_learning using simple deep_learning learning rule."""
        for _ in range(epochs):
            for xs, y in zip(xss, ys):
                pred = self.predict(xs)
                error = y - pred
                self.ws = [w + self.lr * error * x for w, x in zip(self.ws, xs)]
                self.b += self.lr * error
