from typing import List

from mcculloch_and_pitts.gates import OR, AND, NOT


class LatchMemory:
    def __init__(self):
        self.state = 0

    def step(self, external_input: int) -> int:
        # new_state = OR(external_input, state)
        self.state = OR(external_input, self.state)
        return self.state


class FlipFlopMemory:
    def __init__(self):
        self.state = 0

    def step(self, set_input: int, reset_input: int) -> int:
        # state_next = OR(AND(set_input, NOT(reset_input)), AND(self.state, NOT(reset_input)))
        set_and_not_reset = AND(set_input, NOT(reset_input))
        hold_and_not_reset = AND(self.state, NOT(reset_input))
        reset_effect = AND(reset_input, NOT(set_input))
        self.state = OR(set_and_not_reset, hold_and_not_reset)
        self.state = AND(self.state, NOT(reset_effect))  # reset clears
        return self.state


class RegisterMemory:
    def __init__(self, size: int):
        self.cells = [LatchMemory() for _ in range(size)]

    def step(self, inputs: List[int]) -> List[int]:
        return [cell.step(inp) for cell, inp in zip(self.cells, inputs)]

    def read(self) -> List[int]:
        return [cell.state for cell in self.cells]


if __name__ == "__main__":
    print("Latch demo:")
    latch = LatchMemory()
    for t, inp in enumerate([1, 0, 0, 0, 0], 1):
        out = latch.step(inp)
        print(f"t={t}, in={inp}, out={out}")

    print("\nFlipFlop demo (set, reset):")
    ff = FlipFlopMemory()
    for t, (s, r) in enumerate([(1,0), (0,0), (0,1), (0,0), (1,0)], 1):
        out = ff.step(s, r)
        print(f"t={t}, set={s}, reset={r}, out={out}")
