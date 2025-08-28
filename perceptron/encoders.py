import numpy as np

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert integer class labels to one-hot encoding.

    Parameters:
    y: array of shape (batch_size,)
    num_classes: number of classes

    Returns:
    np.ndarray: one-hot encoded labels of shape (batch_size, num_classes)
    """
    return np.eye(num_classes)[y]
