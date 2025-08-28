import math

import numpy as np


def identity(n: float) -> float:
    return n

# Step function returns 0 for inputs <= 0 and 1 for inputs > 0
def step_function(n: float) -> float:
    return 0 if n <= 0 else 1

def sigmoid_function(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

# Leaky ReLU allows a small, non-zero gradient when the unit is not active
def leaky_relu(n: float, alpha: float = 0.01) -> float:
    return max(alpha * n, n)

# Tanh (Hyperbolic Tangent) returns values between -1 and 1
def tanh_function(n: float) -> float:
    return (math.exp(n) - math.exp(-n)) / (math.exp(n) + math.exp(-n))

# ELU (Exponential Linear Unit) has a smooth, non-zero gradient for negative inputs
def elu(n: float, alpha: float = 1.0) -> float:
    return n if n > 0 else alpha * (math.exp(n) - 1)

# Softplus is a smooth approximation to the ReLU function
def softplus(n: float) -> float:
    return math.log(1 + math.exp(n))

# Swish is a smooth, non-monotonic function that has been shown to outperform ReLU in some cases
def swish(n: float, beta: float = 1.0) -> float:
    return n * (1 / (1 + math.exp(-beta * n)))

# GELU (Gaussian Error Linear Unit) is a smooth approximation to the ReLU function
def gelu(n: float) -> float:
    return 0.5 * n * (1 + math.tanh(math.sqrt(2 / math.pi) * (n + 0.044715 * n**3)))

# SELU (Scaled Exponential Linear Unit) allows for self-normalizing neural networks
def selu(n: float) -> float:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    if n > 0:
        return scale * n
    else:
        return scale * alpha * (math.exp(n) - 1)

# Parametric ReLU (PReLU) allows the slope of the negative part to be learned
def prelu(n: float, alpha: float) -> float:
    if n > 0:
        return n
    else:
        return alpha * n

# Maxout is a piecewise linear function that can approximate any convex function
def maxout(n: float, w: list, b: list, k: int) -> float:
    return max(w[i] * n + b[i] for i in range(k))

def softmax_function(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
