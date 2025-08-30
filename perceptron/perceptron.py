import math
from collections.abc import Callable
from typing import List
import random


class Perceptron:
    def __init__(self, input_size: int, activation: Callable[[float], float], learning_rate: float) -> None:
        self.weights = [random.uniform(-1, 1) * math.sqrt(1 / input_size)
                        for _ in range(input_size)]
        self.bias = random.uniform(-0.01, 0.01)
        self.activation: Callable[[float], float] = activation
        self.learning_rate: float = learning_rate

    def predict(self, features: List[float]) -> float:
        return self.activation(
            sum(feature * weight for feature, weight in zip(features, self.weights)) + self.bias
        )

    def fit(self, features_table: List[List[float]], labels: List[float], epochs: int) -> None:
        for _ in range(epochs):
            for features, label in zip(features_table, labels):
                prediction = self.predict(features)
                error_delta = label - prediction
                self.weights = [
                    weight + self.learning_rate * error_delta * feature
                    for weight, feature in zip(self.weights, features)
               ]
                self.bias += self.learning_rate * error_delta
