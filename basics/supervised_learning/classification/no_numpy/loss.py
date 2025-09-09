from typing import List


def mse_loss(outputs: List[float], label: int) -> float:
    targets = [0.0] * len(outputs)
    targets[label] = 1.0
    return sum((o - t) ** 2 for o, t in zip(outputs, targets))
