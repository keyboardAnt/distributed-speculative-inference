import random

import numpy as np


def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
