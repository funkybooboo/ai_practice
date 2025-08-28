from typing import List, Callable

from deep_learning.activation_functions import step_function
from deep_learning.neuron import Neuron


def gate(xs: List[float], ws: List[float], b: float) -> float:
    """Compute gate output using a Perceptron instance."""
    p = Neuron(input_size=len(xs), activation=step_function)
    p.ws = ws[:]  # set weights
    p.b = b       # set bias
    return p.predict(xs)


def print_table(gate: Callable[[float, float], float]) -> None:
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]

    print(f"{'x1':<5} | {'x2':<5} | {f'{gate.__name__}(x1, x2)':<15}")
    print("-" * 25)

    for x1, x2 in inputs:
        result = gate(x1, x2)
        print(f"{x1:<5} | {x2:<5} | {result:<15}")
