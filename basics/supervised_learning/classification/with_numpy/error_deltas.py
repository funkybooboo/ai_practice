import numpy as np


def mse_error_delta(outputs: np.ndarray, y: np.ndarray) -> np.ndarray:
    batch_size, num_classes = outputs.shape
    targets = np.zeros((batch_size, num_classes))
    targets[np.arange(batch_size), y] = 1.0
    return 2 * (outputs - targets) / batch_size
