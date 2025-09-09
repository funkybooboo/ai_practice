import numpy as np


def mse_loss(outputs: np.ndarray, y: np.ndarray) -> float:
    batch_size, num_classes = outputs.shape
    targets = np.zeros((batch_size, num_classes))
    targets[np.arange(batch_size), y] = 1.0
    return float(np.mean(np.square(outputs - targets)))
