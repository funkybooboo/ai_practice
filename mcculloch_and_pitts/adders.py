from typing import List, Tuple
from mcculloch_and_pitts.gates import XOR, AND, OR


def half_adder(x: int, y: int) -> Tuple[int, int]:
    """Half-adder built from gates. Returns (sum, carry)."""
    return XOR(x, y), AND(x, y)


def full_adder(x: int, y: int, carry_in: int) -> Tuple[int, int]:
    """Full-adder built from half-adders and OR. Returns (sum, carry_out)."""
    s1, c1 = half_adder(x, y)
    s2, c2 = half_adder(s1, carry_in)
    carry_out = OR(c1, c2)
    return s2, carry_out


def add_binary(a: List[int], b: List[int]) -> List[int]:
    """
    Adds two binary numbers using neuron-built adders.
    Inputs and outputs are LSB-first lists of bits.
    """
    result: List[int] = []
    carry = 0
    max_len = max(len(a), len(b))

    for i in range(max_len):
        bit_a = a[i] if i < len(a) else 0
        bit_b = b[i] if i < len(b) else 0
        s, carry = full_adder(bit_a, bit_b, carry)
        result.append(s)

    if carry:
        result.append(carry)

    return result


if __name__ == "__main__":
    # Example: 6 (110) + 5 (101) = 11 (1011)
    a = [0, 1, 1]   # 6
    b = [1, 0, 1]   # 5
    result = add_binary(a, b)
    print("Input A (LSB->MSB):", a)
    print("Input B (LSB->MSB):", b)
    print("Result (LSB->MSB):", result)
    print("Decimal result:", int("".join(map(str, reversed(result))), 2))
