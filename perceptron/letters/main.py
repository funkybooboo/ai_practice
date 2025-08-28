from typing import List, Callable

from activation_functions import step_function, identity, sigmoid_function, leaky_relu, relu, tanh_function, softplus, elu, gelu, selu
from flatteners import flatten
from perceptron import perceptron, train_perceptron


def read_letter_file(file_path: str) -> List[List[float]]:
    letter: List[List[float]] = []
    with open(file_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            if line.startswith('#'):
                continue
            cs = []
            for c in line:
                if c == " ":
                    continue
                cs.append(float(c))
            letter.append(cs)
    return letter


def main():
    left_t: List[float] = flatten(read_letter_file('./t/left_t.txt'))
    right_t: List[float] = flatten(read_letter_file('./t/right_t.txt'))
    left_j: List[float] = flatten(read_letter_file('./j/left_j.txt'))
    right_j: List[float] = flatten(read_letter_file('./j/right_j.txt'))

    _as: List[Callable[[float], float]] = [
        identity, step_function, sigmoid_function, relu, leaky_relu,
        tanh_function, elu, softplus, gelu, selu
    ]

    lrs: List[float] = [0.1, 1, 10]
    bs: List[float] = [-10, -1, 0, 1, 10]
    epochs_list: List[int] = [10, 100, 1000]

    for a in _as:
        for lr in lrs:
            for b in bs:
                for epochs in epochs_list:
                    ws: List[float] = [0 for _ in range(len(left_t))]

                    # Train perceptron
                    xss = [left_t, right_t, left_j, right_j]
                    ys = [1, 1, 0, 0]  # Labeling: 1 for 't', 0 for 'j'
                    ws, b = train_perceptron(xss, ys, ws, b, a, lr, epochs)

                    # After training, test the perceptron on the examples
                    left_t_r: float = perceptron(left_t, ws, b, a)
                    right_t_r: float = perceptron(right_t, ws, b, a)
                    left_j_r: float = perceptron(left_j, ws, b, a)
                    right_j_r: float = perceptron(right_j, ws, b, a)

                    print("Can a perceptron tell the difference between a t and a j?")
                    print("Activation function:", a.__name__)
                    print("Learning rate:", lr)
                    print("Bias:", b)
                    print("Epochs:", epochs)
                    print("t >= 0.5, j < 0.5")
                    print()
                    print("left_t", left_t_r)
                    print("right_t", right_t_r)
                    print("left_j", left_j_r)
                    print("right_j", right_j_r)
                    print()
                    print("---")
                    print()


if __name__ == "__main__":
    main()
