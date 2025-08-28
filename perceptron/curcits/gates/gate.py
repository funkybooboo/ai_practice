from typing import List, Callable

from activation_functions import step_function
from perceptron import perceptron


def gate(xs: List[float], ws: List[float], b: float) -> float:
    return perceptron(xs, ws, b, step_function)


def print_table(gate: Callable[[float, float], float]) -> None:
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]

    print(f"{'x1':<5} | {'x2':<5} | {f'{gate.__name__}(x1, x2)':<15}")
    print("-" * 25)

    for x1, x2 in inputs:
        result = gate(x1, x2)
        print(f"{x1:<5} | {x2:<5} | {result:<15}")
