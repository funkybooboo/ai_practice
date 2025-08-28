from typing import List, Tuple

from deep_learning.curcits.gates.and_gate import and_gate
from deep_learning.curcits.gates.not_gate import not_gate
from deep_learning.curcits.gates.or_gate import or_gate


def half_adder(a: float, b: float) -> Tuple[float, float]:
    _sum = or_gate(and_gate(a, not_gate(b)), and_gate(not_gate(a), b))
    carry = and_gate(a, b)
    return _sum, carry

def full_adder(a: float, b: float, cin: float) -> Tuple[float, float]:
    sum1, carry1 = half_adder(a, b)
    sum2, carry2 = half_adder(sum1, cin)
    carry = or_gate(carry1, carry2)
    return sum2, carry

def one_bit_adder(a: float, b: float, cin: float = 0) -> Tuple[float, float]:
    return full_adder(a, b, cin)

def four_bit_adder(a: List[float], b: List[float], cin: float = 0) -> Tuple[List[float], float]:
    if len(a) != 4 or len(b) != 4:
        raise ValueError("Inputs must be 4-bit lists")

    sum_bits = []
    carry = cin
    for i in range(4):
        sum_bit, carry = one_bit_adder(a[i], b[i], carry)
        sum_bits.append(sum_bit)

    return sum_bits, carry


if __name__ == "__main__":
    print("1-Bit Adder:")
    print("Input (a, b, cin): (0, 0, 0) -> Sum, Carry:", one_bit_adder(0, 0, 0))
    print("Input (a, b, cin): (0, 1, 0) -> Sum, Carry:", one_bit_adder(0, 1, 0))
    print("Input (a, b, cin): (1, 0, 0) -> Sum, Carry:", one_bit_adder(1, 0, 0))
    print("Input (a, b, cin): (1, 1, 0) -> Sum, Carry:", one_bit_adder(1, 1, 0))
    print("Input (a, b, cin): (0, 0, 1) -> Sum, Carry:", one_bit_adder(0, 0, 1))
    print("Input (a, b, cin): (0, 1, 1) -> Sum, Carry:", one_bit_adder(0, 1, 1))
    print("Input (a, b, cin): (1, 0, 1) -> Sum, Carry:", one_bit_adder(1, 0, 1))
    print("Input (a, b, cin): (1, 1, 1) -> Sum, Carry:", one_bit_adder(1, 1, 1))
    print()
    print("4-Bit Adder:")
    print("Input (a, b, cin): (0, 0, 0, 0), (0, 0, 0, 0), 0 -> Sum, Carry:", four_bit_adder([0, 0, 0, 0], [0, 0, 0, 0], 0))
    print("Input (a, b, cin): (0, 0, 0, 1), (0, 0, 0, 0), 0 -> Sum, Carry:", four_bit_adder([0, 0, 0, 1], [0, 0, 0, 0], 0))
    print("Input (a, b, cin): (0, 0, 1, 0), (0, 0, 0, 1), 0 -> Sum, Carry:", four_bit_adder([0, 0, 1, 0], [0, 0, 0, 1], 0))
    print("Input (a, b, cin): (1, 1, 1, 1), (0, 0, 0, 0), 0 -> Sum, Carry:", four_bit_adder([1, 1, 1, 1], [0, 0, 0, 0], 0))
    print("Input (a, b, cin): (1, 1, 1, 1), (1, 1, 1, 1), 0 -> Sum, Carry:", four_bit_adder([1, 1, 1, 1], [1, 1, 1, 1], 0))
    print("Input (a, b, cin): (1, 1, 1, 1), (1, 1, 1, 1), 1 -> Sum, Carry:", four_bit_adder([1, 1, 1, 1], [1, 1, 1, 1], 1))
