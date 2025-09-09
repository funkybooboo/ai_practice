from typing import List


def diff_error_delta(outputs: List[float], label: int) -> List[float]:
    targets = [0.0] * len(outputs)
    targets[label] = 1.0
    return [o - t for o, t in zip(outputs, targets)]
