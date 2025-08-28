from typing import List, Callable

from deep_learning.activation_functions import (
    step_function, identity, sigmoid_function, leaky_relu, relu, tanh_function,
    softplus, elu, gelu, selu
)
from deep_learning.flatteners import flatten
from deep_learning.neuron import Neuron


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

    activations: List[Callable[[float], float]] = [
        identity, step_function, sigmoid_function, relu, leaky_relu,
        tanh_function, elu, softplus, gelu, selu
    ]

    learning_rates: List[float] = [0.1, 1, 10]
    biases: List[float] = [-10, -1, 0, 1, 10]
    epochs_list: List[int] = [10, 100, 1000]

    for activation in activations:
        for lr in learning_rates:
            for bias in biases:
                for epochs in epochs_list:
                    # Initialize Perceptron
                    p = Neuron(input_size=len(left_t), activation=activation, lr=lr)
                    p.b = bias  # set initial bias

                    # Train deep_learning
                    xss = [left_t, right_t, left_j, right_j]
                    ys = [1.0, 1.0, 0.0, 0.0]  # 1 for 't', 0 for 'j'
                    p.fit(xss, ys, epochs=epochs)

                    # Test deep_learning
                    left_t_r = p.predict(left_t)
                    right_t_r = p.predict(right_t)
                    left_j_r = p.predict(left_j)
                    right_j_r = p.predict(right_j)

                    print("Can a deep_learning tell the difference between a t and a j?")
                    print("Activation function:", activation.__name__)
                    print("Learning rate:", lr)
                    print("Bias:", p.b)
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
