def sigmoid_derivative(y: float) -> float:
    return y * (1 - y)

def relu_derivative(y: float) -> float:
    return 1.0 if y > 0 else 0.0
