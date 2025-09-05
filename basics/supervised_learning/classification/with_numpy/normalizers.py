import numpy as np

def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize an image (any shape) to values between 0 and 1.
    """
    image = np.asarray(image, dtype=np.float32)
    return image / 255.0
