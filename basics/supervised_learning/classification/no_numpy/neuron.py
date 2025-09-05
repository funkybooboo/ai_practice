import random
from typing import Callable, List


class Neuron:
    def __init__(self, input_size: int, activation: Callable[[float], float],
                 activation_derivative: Callable[[float], float], learning_rate: float):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(input_size)]
        self.bias = 0
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.learning_rate = learning_rate
        self.last_features = []
        self.last_output = 0

    def forward(self, features: List[float]) -> float:
        self.last_features = features
        self.last_output = self.activation(
            sum(feature * weight for feature, weight in zip(features, self.weights)) + self.bias
        )
        return self.last_output

    def backward(self, error_delta: float):
        error_gradiant = error_delta * self.activation_derivative(self.last_output)

        self.weights = [
            weight - self.learning_rate * error_gradiant * feature
            for weight, feature in zip(self.weights, self.last_features)
        ]
        self.bias -= self.learning_rate * error_gradiant

        return [error_gradiant * w for w in self.weights]
