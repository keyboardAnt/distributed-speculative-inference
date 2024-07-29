import multiprocessing
import time
from unittest.mock import patch

import pytest

from dsi.configs.experiment.simul.online import ConfigDSIOnline, SimulType
from dsi.online.simul.simul import SimulOnline, restart_draft


@pytest.fixture(
    params=[
        {
            "c": 0.05725204603746534,
            "a": 0.94,
            "k": 5,
            "failure_cost": 0.18028411593660712,
            "S": 50,
            "num_repeats": 30,
            "num_target_servers": 4,
            "c_sub": 0.0033411221550777503,
            "failure_cost_sub": 0.10018306512758136,
            "total_tokens": 100,
            "wait_for_pipe": 0.1,
            "simul_type": SimulType.DSI,
            "maximum_correct_tokens": 20,
        },
        {"S": 49, "num_repeats": 1},
        {"a": 0.01, "S": 49, "num_repeats": 1},
    ]
)
def config(request):
    # Create a ConfigDSIOnline instance with the parameters provided by each param set
    return ConfigDSIOnline(**request.param)


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


@pytest.fixture
def latency_min(config: ConfigDSIOnline) -> float:
    return config.num_repeats * config.S * config.c_sub


@pytest.fixture
def latency_max(config: ConfigDSIOnline) -> float:
    # Calculated within the fixture, ensuring correct and dynamic evaluation
    return (
        config.num_repeats
        * (config.total_tokens + config.S)
        * (config.failure_cost + config.wait_for_pipe)
        * 1.5
    )


@pytest.mark.skip(reason="#37")
@pytest.mark.timeout(90)
@pytest.mark.online
def test_duration(config: ConfigDSIOnline, latency_min: float, latency_max: float):
    """
    Execute the experiment with the default configuration. Validate that the duration
    is within a reasonable range.
    """
    start = time.time()
    SimulOnline(config).run()
    end = time.time()
    duration = end - start
    assert (
        latency_min <= duration <= latency_max
    ), f"Duration {duration} out of expected range ({latency_min}, {latency_max})"


@pytest.mark.skip(reason="#37")
@pytest.mark.timeout(90)
@pytest.mark.online
def test_num_of_fix_history(config: ConfigDSIOnline):
    """
    Execute the experiment and validate that the number of calls to the `fix_history`
    function matches the expected number of calls.
    """
    config.num_repeats = 1
    with patch("dsi.online.simul.core.fix_history") as mock_fix_history:
        SimulOnline(config).run()
        assert (
            mock_fix_history.call_count <= config.S
        ), f"Number of calls to fix_history exceeds the expected value ({config.S})"
