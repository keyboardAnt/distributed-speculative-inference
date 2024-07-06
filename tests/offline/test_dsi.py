import numpy as np
import pytest

from dsi.configs.run.run import ConfigRunDSI
from dsi.offline.run.dsi import RunDSI
from dsi.offline.run.si import RunSI
from dsi.types.exception import (
    DrafterSlowerThanTargetError,
    NumOfTargetServersInsufficientError,
)
from dsi.types.result import Result


def test_dsi_result_shapes():
    num_repeats: int = 17
    config = ConfigRunDSI(num_repeats=num_repeats)
    dsi = RunDSI(config)
    res: Result = dsi.run()
    assert (
        len(res.cost_per_run) == num_repeats
    ), f"expected {num_repeats} results, got {len(res.cost_per_run)}"


@pytest.mark.parametrize("c", [0.01, 0.1, 0.5, 0.9, 0.99, 1.0, 1.01, 2.0, 1000])
@pytest.mark.parametrize("failure_cost", [0.01, 0.1, 0.5, 0.9, 0.99, 1.0, 10, 1000])
@pytest.mark.parametrize("a", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
@pytest.mark.parametrize("k", [1, 10, 100, 1000, 100000000])
def test_dsi_faster_than_si_and_nonsi(c: float, failure_cost: float, a: float, k: int):
    try:
        config = ConfigRunDSI(c=c, failure_cost=failure_cost, a=a, k=k)
    except (NumOfTargetServersInsufficientError, DrafterSlowerThanTargetError):
        return
    si = RunSI(config)
    dsi = RunDSI(config)
    si_res: Result = si.run()
    dsi_res: Result = dsi.run()
    num_iterations_min: int = config.S // (config.k + 1)
    for res in [si_res, dsi_res]:
        for num_iters in res.num_iters_per_run:
            assert num_iters >= num_iterations_min
    for cost_si, cost_dsi in zip(si_res.cost_per_run, dsi_res.cost_per_run):
        assert cost_dsi <= cost_si or np.isclose(
            cost_si, cost_dsi
        ), f"DSI is never slower than SI. DSI: {cost_dsi}, SI: {cost_si}"
        cost_min: float = (num_iterations_min - 1) * config.c + config.failure_cost
        cost_max: float = config.S * (config.c * config.k + config.failure_cost)
        assert cost_min <= cost_dsi or np.isclose(cost_min, cost_dsi)
        assert cost_dsi <= cost_max or np.isclose(cost_max, cost_dsi)
    dsi_cost_per_run_arr: np.ndarray = np.array(dsi_res.cost_per_run)
    cost_nonsi: float = config.S * config.failure_cost
    diff: np.ndarray = dsi_cost_per_run_arr - cost_nonsi
    assert (diff <= 0).all() or np.isclose(diff[diff > 0], 0).all(), (
        "DSI is never slower than non-SI."
        f" DSI: {dsi_cost_per_run_arr}, non-SI: {cost_nonsi}"
    )
