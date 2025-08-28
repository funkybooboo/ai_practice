from collections.abc import Callable
from typing import List


def flatten_xs(raw_xs: List[List[float]]) -> List[float]:
    xs: List[float] = []
    for row in raw_xs:
        for c in row:
            xs.append(c)
    return xs


def perceptron(xs: List[float], ws: List[float], b: float, a: Callable[[float], float]) -> float:
    s = 0
    for x, w in zip(xs, ws):
        s += x * w
    s += b
    return a(s)


def train_perceptron(xss: List[List[float]], ys: List[int], ws: List[float], b: float, a: Callable[[float], float], lr: float, epochs: int) -> (List[float], float):
    for _ in range(epochs):
        for xs, y in zip(xss, ys):
            prediction = perceptron(xs, ws, b, a)
            error = y - prediction
            ws = [w + lr * error * x for x, w in zip(xs, ws)]
            b += lr * error
    return ws, b
