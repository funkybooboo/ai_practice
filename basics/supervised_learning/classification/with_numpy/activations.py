import numpy as np


def step(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)  # numerically stable


def swish(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    return x / (1 + np.exp(-beta * x))


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def selu(x: np.ndarray) -> np.ndarray:
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    return np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))


def prelu(x: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)


def maxout(x: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Maxout activation.
    x: (batch, input_size)
    W: (k, input_size)
    b: (k,)
    Returns: (batch,)
    """
    z = x @ W.T + b  # shape (batch, k)
    return np.max(z, axis=1)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax, stable version"""
    z = z - np.max(z, axis=1, keepdims=True)  # stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
