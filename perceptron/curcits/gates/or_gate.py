from perceptron.curcits.gates.gate import gate, print_table


def or_gate(x1: float, x2: float) -> float:
    return gate([x1, x2], [2, 2], -1)


if __name__ == "__main__":
    print_table(or_gate)
