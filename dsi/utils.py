import random

import numpy as np


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_arange(start, stop, step) -> np.ndarray:
    """
    Return evenly spaced values within a given interval. This function is similar to
    numpy.arange, but with safe floating point arithmetic.
    Reference: https://stackoverflow.com/a/47250077
    """
    return step * np.arange(start / step, stop / step)
