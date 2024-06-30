import numpy as np


def generate_num_accepted_drafts(
    acceptance_rate: float, lookahead: int, max_num_samples: int
):
    """
    Generator that samples the number of accepted draft tokens in SI or
    DSI using the specified acceptance rate and lookahead. It samples
    S times at once and then yields each sample one by one.

    :param acceptance_rate: The rate of acceptance for draft tokens.
    :param lookahead: The maximum possible number of accepted drafts.
    :param S: The number of samples to generate.
    """
    if acceptance_rate == 0:
        samples = [0] * max_num_samples
    elif acceptance_rate == 1:
        samples = [lookahead] * max_num_samples
    else:
        # Sample S values from the geometric distribution and adjust them
        samples = np.random.geometric(1 - acceptance_rate, size=max_num_samples) - 1
        # Ensure no value exceeds the lookahead limit
        samples = np.minimum(samples, lookahead)

    # Yield each sample one by one
    yield from samples
