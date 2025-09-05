import numpy as np


def sigmoid_derivative(y: np.ndarray) -> np.ndarray:
    """y is already sigmoid(x)"""
    return y * (1 - y)


def relu_derivative(y: np.ndarray) -> np.ndarray:
    """y is already relu(x)"""
    return (y > 0).astype(float)
