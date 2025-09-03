import math
from typing import List


def step(n: float) -> float:
    return 0 if n <= 0 else 1


def sigmoid(n: float) -> float:
    return 1 / (1 + math.exp(-n))


def relu(n: float) -> float:
    return max(0, n)


def leaky_relu(n: float, alpha: float = 0.01) -> float:
    return max(alpha * n, n)


def tanh(n: float) -> float:
    return (math.exp(n) - math.exp(-n)) / (math.exp(n) + math.exp(-n))


def elu(n: float, alpha: float = 1.0) -> float:
    return n if n > 0 else alpha * (math.exp(n) - 1)


def softplus(n: float) -> float:
    return math.log(1 + math.exp(n))


def swish(n: float, beta: float = 1.0) -> float:
    return n * (1 / (1 + math.exp(-beta * n)))


def gelu(n: float) -> float:
    return 0.5 * n * (1 + math.tanh(math.sqrt(2 / math.pi) * (n + 0.044715 * n**3)))


def selu(n: float) -> float:
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * n if n > 0 else scale * alpha * (math.exp(n) - 1)


def prelu(n: float, alpha: float) -> float:
    return n if n > 0 else alpha * n


def maxout(n: float, w: List[float], b: List[float], k: int) -> float:
    return max(w[i] * n + b[i] for i in range(k))


def softmax(z: List[float]) -> List[float]:
    max_z = max(z)  # stability
    exp_z = [math.exp(val - max_z) for val in z]
    sum_exp = sum(exp_z)
    return [val / sum_exp for val in exp_z]
