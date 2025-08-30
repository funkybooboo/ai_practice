from perceptron.activations import step
from perceptron.perceptron import Perceptron

# ---- Basic gates ----
def AND(x1: int, x2: int) -> int:
    p = Perceptron(input_size=2, activation=step, learning_rate=0.1)
    p.weights = [1.0, 1.0]
    p.bias = -1.5
    return int(p.predict([x1, x2]))

def OR(x1: int, x2: int) -> int:
    p = Perceptron(input_size=2, activation=step, learning_rate=0.1)
    p.weights = [1.0, 1.0]
    p.bias = -0.5
    return int(p.predict([x1, x2]))

def NOT(x: int) -> int:
    p = Perceptron(input_size=1, activation=step, learning_rate=0.1)
    p.weights = [-1.0]
    p.bias = 0.5
    return int(p.predict([x]))

# ---- Derived gates ----
def XOR(x: int, y: int) -> int:
    # (x OR y) AND NOT(x AND y)
    return AND(OR(x, y), NOT(AND(x, y)))

def NAND(x: int, y: int) -> int:
    return NOT(AND(x, y))

def NOR(x: int, y: int) -> int:
    return NOT(OR(x, y))

def XNOR(x: int, y: int) -> int:
    return NOT(XOR(x, y))

def IMPLIES(x: int, y: int) -> int:
    # x -> y == (NOT x) OR y
    return OR(NOT(x), y)

def BUFFER(x: int) -> int:
    p = Perceptron(input_size=1, activation=step, learning_rate=0.1)
    p.weights = [1.0]
    p.bias = 0.0
    return int(p.predict([x]))

def NIMPLIES(x1: int, x2: int) -> int:
    return AND(x1, NOT(x2))

def EQUIV(x1: int, x2: int) -> int:
    return XNOR(x1, x2)

def DIFF(x1: int, x2: int) -> int:
    return AND(x1, NOT(x2))

if __name__ == "__main__":
    tests = [(0,0), (0,1), (1,0), (1,1)]
    print("x y | AND OR NAND NOR XOR XNOR IMPLIES")
    print("--------------------------------------")
    for x, y in tests:
        print(f"{x} {y} |  {AND(x,y)}   {OR(x,y)}    {NAND(x,y)}    {NOR(x,y)}    {XOR(x,y)}     {XNOR(x,y)}      {IMPLIES(x,y)}")
