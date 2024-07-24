import multiprocessing
import time

import pytest

from dsi.configs.experiment.simul.online import ConfigDSIOnline, SimulType
from dsi.online.simul.simul import SimulOnline, restart_draft


@pytest.fixture
def config():
    return ConfigDSIOnline(
        c=0.05725204603746534,
        a=0.94,
        k=5,
        failure_cost=0.18028411593660712,
        S=50,
        num_repeats=30,
        num_target_servers=4,
        c_sub=0.0033411221550777503,
        failure_cost_sub=0.10018306512758136,
        total_tokens=100,
        wait_for_pipe=0.1,
        simul_type=SimulType.DSI,
        maximum_correct_tokens=20,
    )


def test_entire_threadpool_used(config: ConfigDSIOnline):
    total_tokens = config.total_tokens
    sim_shared_dict = multiprocessing.Manager().dict()

    sim_shared_dict["total_tokens"] = total_tokens
    sim_shared_dict["prompt_tokens"] = total_tokens

    # sample number of correct tokens
    sim_shared_dict["correct"] = 20

    # While the stop signal is not received, keep restarting the draft model
    th = restart_draft(
        config, sim_shared_dict["total_tokens"], sim_shared_dict, config.wait_for_pipe
    )
    th.join()

    for i in range(config.num_target_servers):
        assert str(i) in sim_shared_dict


def test_single_thread_in_si(config: ConfigDSIOnline):
    config.simul_type = SimulType.SI

    total_tokens = config.total_tokens
    sim_shared_dict = multiprocessing.Manager().dict()

    sim_shared_dict["total_tokens"] = total_tokens
    sim_shared_dict["prompt_tokens"] = total_tokens

    # sample number of correct tokens
    sim_shared_dict["correct"] = 20

    # While the stop signal is not received, keep restarting the draft model
    th = restart_draft(
        config, sim_shared_dict["total_tokens"], sim_shared_dict, config.wait_for_pipe
    )
    th.join()

    assert "MainThread" in sim_shared_dict


def test_correct_token_count_per_iteration(config: ConfigDSIOnline):
    correct_token_list = [5, 15, 3, 7, 10, 5, 20]

    total_tokens = config.total_tokens
    sim_shared_dict = multiprocessing.Manager().dict()

    sim_shared_dict["total_tokens"] = total_tokens
    sim_shared_dict["prompt_tokens"] = total_tokens

    iter_till_stop = 0

    # While the stop signal is not received, keep restarting the draft model
    while "stop" not in sim_shared_dict:
        # assert that the number of tokens is in accordance with the accepted token list
        assert (
            sim_shared_dict["total_tokens"]
            == sum(correct_token_list[:iter_till_stop]) + config.total_tokens
        )

        # sample number of correct tokens
        sim_shared_dict["correct"] = correct_token_list[iter_till_stop]

        th = restart_draft(
            config,
            sim_shared_dict["total_tokens"],
            sim_shared_dict,
            config.wait_for_pipe,
        )
        th.join()
        iter_till_stop += 1


@pytest.fixture
def config_simple():
    return ConfigDSIOnline(num_repeats=1, S=90)


@pytest.fixture
def latency_min(config_simple: ConfigDSIOnline) -> float:
    return config_simple.num_repeats * config_simple.S * config_simple.c_sub


@pytest.fixture
def latency_max(config_simple: ConfigDSIOnline) -> float:
    # Calculated within the fixture, ensuring correct and dynamic evaluation
    return (
        config_simple.num_repeats
        * (config_simple.total_tokens + config_simple.S)
        * (config_simple.failure_cost + config_simple.wait_for_pipe)
        * 1.5
    )


@pytest.mark.timeout(100)
def test_duration(
    config_simple: ConfigDSIOnline, latency_min: float, latency_max: float
):
    """
    Execute the experiment with the default configuration. Validate that the duration
    is within a reasonable range.
    """

    start = time.time()
    SimulOnline(config_simple).run()
    end = time.time()
    duration = end - start
    assert (
        latency_min <= duration <= latency_max
    ), f"Duration {duration} out of expected range ({latency_min}, {latency_max})"
