def normalize_image(flat_image: list[float]) -> list[float]:
    """
    Normalize a 1D image to values between 0 and 1.
    """
    return [pixel / 255.0 for pixel in flat_image]
