from deep_learning.curcits.gates.gate import gate, print_table


def and_gate(x1: float, x2: float) -> float:
    return gate([x1, x2], [2, 2], -3)


if __name__ == "__main__":
    print_table(and_gate)
