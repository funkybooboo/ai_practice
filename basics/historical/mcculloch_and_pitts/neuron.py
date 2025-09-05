from typing import List


class Neuron:
    def __init__(self, weights: List[int], threshold: int):
        self.weights = weights
        self.threshold = threshold
        self.output: int = 0  # 0 or 1 only

    def activate(self, inputs: List[int]) -> int:
        total = sum(w * i for w, i in zip(self.weights, inputs))
        self.output = 1 if total >= self.threshold else 0
        return self.output
