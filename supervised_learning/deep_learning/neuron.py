import math
import random
from typing import Callable, List


class Neuron:
    def __init__(self, input_size: int, activation: Callable[[float], float],
                 activation_derivative: Callable[[float], float], learning_rate: float, is_output_neuron=False):
        self.weights = [random.uniform(-1, 1) * math.sqrt(1 / input_size) for _ in range(input_size)]
        self.bias = random.uniform(-0.01, 0.01)
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.learning_rate = learning_rate
        self.last_features = []
        self.last_output = 0
        self.is_output_neuron = is_output_neuron

    def forward(self, features: List[float]) -> float:
        self.last_features = features
        self.last_output = self.activation(
            sum(feature * weight for feature, weight in zip(features, self.weights)) + self.bias
        )
        return self.last_output

    def backward(self, error_delta: float):
        error_gradiant = error_delta if self.is_output_neuron else error_delta * self.activation_derivative(self.last_output)

        self.weights = [
            weight - self.learning_rate * error_gradiant * feature
            for weight, feature in zip(self.weights, self.last_features)
        ]
        self.bias -= self.learning_rate * error_gradiant

        return [error_gradiant * w for w in self.weights]
