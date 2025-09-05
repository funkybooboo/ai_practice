import numpy as np
from typing import Any

def flatten(arr: Any) -> np.ndarray:
    """
    Flattens a nested list or numpy array into a 1D numpy array.
    """
    return np.array(arr).ravel()
