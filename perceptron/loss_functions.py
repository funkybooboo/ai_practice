import numpy as np

def cross_entropy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Categorical cross-entropy loss.

    Parameters:
    y_pred: predicted probabilities, shape (batch_size, num_classes)
    y_true: one-hot encoded true labels, shape (batch_size, num_classes)

    Returns:
    float: average cross-entropy loss over the batch
    """
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m
