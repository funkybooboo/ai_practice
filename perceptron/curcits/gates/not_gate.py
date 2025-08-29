from deep_learning.curcits.gates.gate import gate


def not_gate(x: float) -> float:
    return gate([x], [-1], 0.5)


if __name__ == "__main__":
    inputs = [0, 1]

    # Print the table header
    print(f"{'x':<5} | {'NOT(x)':<15}")
    print("-" * 20)

    for x in inputs:
        result = not_gate(x)
        print(f"{x:<5} | {result:<15}")
