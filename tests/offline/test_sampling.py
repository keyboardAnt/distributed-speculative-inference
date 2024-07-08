import time
from typing import Generator

import numpy as np

from dsi.offline.run.common import generate_num_accepted_drafts
from dsi.utils import set_random_seed


def test_get_num_accepted_tokens_acceptance_rate_zero():
    lookaheads: list[int] = [0, 1, 2, 10, 1000]
    latencies: list[float] = []
    for lookahead in lookaheads:
        start_time: float = time.time()
        sampler: Generator = generate_num_accepted_drafts(
            acceptance_rate=0, lookahead=lookahead, max_num_samples=1000
        )
        num_accepted_tokens: int = next(sampler)
        end_time: float = time.time()
        assert (
            num_accepted_tokens == 0
        ), "For an acceptance rate of 0, the number of accepted tokens must be 0."
        latencies.append(end_time - start_time)
    latencies_array: np.ndarray = np.array(latencies)
    assert np.allclose(latencies_array, latencies_array.mean(), atol=1e-2), (
        "The latency of sampling the number of accepted tokens should not depend on"
        " the lookahead."
    )


def test_get_num_accepted_tokens_acceptance_rate_one():
    lookaheads: list[int] = [0, 1, 2, 10, 1000]
    latencies: list[float] = []
    for lookahead in lookaheads:
        start_time: float = time.time()
        sampler: Generator = generate_num_accepted_drafts(
            acceptance_rate=1, lookahead=lookahead, max_num_samples=1000
        )
        num_accepted_tokens: int = next(sampler)
        end_time: float = time.time()
        assert num_accepted_tokens == lookahead, (
            "For an acceptance rate of 1, the number of accepted tokens must be equal"
            " to the lookahead."
        )
        latencies.append(end_time - start_time)
    latencies_array = np.array(latencies)
    assert np.allclose(latencies_array, latencies_array.mean(), atol=1e-2), (
        "The latency of sampling the number of accepted tokens should not depend on"
        " the lookahead."
    )


def test_get_num_accepted_tokens_acceptance_rate_random():
    lookaheads: list[int] = [0, 1, 2, 10, 1000]
    acceptance_rates: list[float] = [0.1, 0.5, 0.9]
    latencies: list[float] = []
    for acceptance_rate in acceptance_rates:
        for lookahead in lookaheads:
            start_time: float = time.time()
            sampler: Generator = generate_num_accepted_drafts(
                acceptance_rate=acceptance_rate,
                lookahead=lookahead,
                max_num_samples=1000,
            )
            num_accepted_tokens: int = next(sampler)
            end_time: float = time.time()
            assert 0 <= num_accepted_tokens <= lookahead, (
                "For any acceptance rate, the number of accepted tokens must be in the"
                " range [0, lookahead]."
            )
            latencies.append(end_time - start_time)
    latencies_array = np.array(latencies)
    assert np.allclose(latencies_array, latencies_array.mean(), atol=1e-2), (
        "The latency of sampling the number of accepted tokens should not depend on"
        " the acceptance rate or lookahead."
    )


def test_samplers_alignment():
    num_samplers = 5000  # Reduced for debugging
    S = 10
    samplers = []
    print("Testing with seed reset:")
    for i in range(num_samplers):
        set_random_seed(0)
        sampler = generate_num_accepted_drafts(
            acceptance_rate=0.5, lookahead=10, max_num_samples=S
        )
        samplers.append(sampler)
        print(
            f"Sampler {i}: First item: {next(sampler)}"
        )  # Print first item to check initial sequence

    all_samples = [list(sampler) for sampler in samplers]
    assert all(
        samples == all_samples[0] for samples in all_samples[1:]
    ), "With seed reset, all samplers should yield the same samples."

    print("Testing without seed reset:")
    samplers = [
        generate_num_accepted_drafts(
            acceptance_rate=0.5, lookahead=10, max_num_samples=S
        )
        for _ in range(num_samplers)
    ]
    all_samples = [list(sampler) for sampler in samplers]
    assert any(
        samples != all_samples[0] for samples in all_samples[1:]
    ), "Without seed reset, samplers should yield different samples."
