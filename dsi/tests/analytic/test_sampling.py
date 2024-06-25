import time

import numpy as np

from dsi.analytic.common import get_num_accepted_tokens


def test_get_num_accepted_tokens_acceptance_rate_zero() -> None:
    lookaheads: list[int] = [0, 1, 2, 10, 1000]
    latencies: list[float] = []
    for lookahead in lookaheads:
        start_time: float = time.time()
        num_accepted_tokens: int = get_num_accepted_tokens(
            acceptance_rate=0, lookahead=lookahead
        )
        end_time: float = time.time()
        assert (
            num_accepted_tokens == 0
        ), "For an acceptance rate of 0, the number of accepted tokens must be 0."
        latencies.append(end_time - start_time)
    latencies_array: np.ndarray = np.array(latencies)
    assert np.allclose(
        latencies_array, latencies_array.mean(), atol=1e-2
    ), "The latency of sampling the number of accepted tokens should not depend on the lookahead."


def test_get_num_accepted_tokens_acceptance_rate_one() -> None:
    lookaheads: list[int] = [0, 1, 2, 10, 1000]
    latencies: list[float] = []
    for lookahead in lookaheads:
        start_time: float = time.time()
        num_accepted_tokens: int = get_num_accepted_tokens(
            acceptance_rate=1, lookahead=lookahead
        )
        end_time: float = time.time()
        assert (
            num_accepted_tokens == lookahead
        ), "For an acceptance rate of 1, the number of accepted tokens must be equal to the lookahead."
        latencies.append(end_time - start_time)
    latencies_array = np.array(latencies)
    assert np.allclose(
        latencies_array, latencies_array.mean(), atol=1e-2
    ), "The latency of sampling the number of accepted tokens should not depend on the lookahead."


def test_get_num_accepted_tokens_acceptance_rate_random() -> None:
    lookaheads: list[int] = [0, 1, 2, 10, 1000]
    acceptance_rates: list[float] = [0.1, 0.5, 0.9]
    latencies: list[float] = []
    for acceptance_rate in acceptance_rates:
        for lookahead in lookaheads:
            start_time: float = time.time()
            num_accepted_tokens: int = get_num_accepted_tokens(
                acceptance_rate=acceptance_rate, lookahead=lookahead
            )
            end_time: float = time.time()
            assert (
                0 <= num_accepted_tokens <= lookahead
            ), "For any acceptance rate, the number of accepted tokens must be in the range [0, lookahead]."
            latencies.append(end_time - start_time)
    latencies_array = np.array(latencies)
    assert np.allclose(
        latencies_array, latencies_array.mean(), atol=1e-2
    ), "The latency of sampling the number of accepted tokens should not depend on the acceptance rate or lookahead."
