from typing import List

from perceptron.curcits.adders import add_binary
from perceptron.curcits.memory import RegisterMemory


class FibonacciBinaryMachine:
    def __init__(self, size: int = 8):
        self.prev = RegisterMemory(size)  # fib(n-1)
        self.curr = RegisterMemory(size)  # fib(n)
        self.size = size

        # Initialize with fib(0)=0, fib(1)=1
        self.prev.step([0] * size)
        self.curr.step([1] + [0] * (size - 1))  # binary "1"

    def step(self) -> List[int]:
        a = self.prev.read()
        b = self.curr.read()
        nxt = add_binary(a, b)[:self.size]

        # Shift registers
        self.prev.step(b)
        self.curr.step(nxt)

        return nxt

    def read(self) -> List[int]:
        return self.curr.read()


if __name__ == "__main__":
    fib = FibonacciBinaryMachine(size=8)

    print("Fibonacci sequence in 8-bit binary (LSB first):")
    for i in range(10):
        val = fib.read()
        print(f"n={i+1}: {val[::-1]}  (binary MSB->LSB)")
        fib.step()
