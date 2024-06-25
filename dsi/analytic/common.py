import numpy as np


def get_num_new_tokens(acceptance_rate: float, lookahead: int) -> int:
    """
    Sample the number of accepted tokens (in SI or DSI) and adds one. The number is in the range [1, inf).
    """
    if acceptance_rate == 0:
        return 0
    if acceptance_rate == 1:
        return lookahead
    return np.random.geometric(1 - acceptance_rate)
