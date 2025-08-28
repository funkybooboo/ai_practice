import math
from typing import List, Tuple


def zeros(shape: Tuple[int, int]) -> List[List[float]]:
    return [[0.0 for _ in range(shape[1])] for _ in range(shape[0])]


def argmax_2d(matrix: List[List[float]]) -> List[int]:
    return [max(range(len(row)), key=lambda i: row[i]) for row in matrix]


def softmax_list(X: List[List[float]]) -> List[List[float]]:
    result = []
    for row in X:
        max_val = max(row)
        exps = [math.exp(x - max_val) for x in row]
        sum_exps = sum(exps)
        result.append([x / sum_exps for x in exps])
    return result
