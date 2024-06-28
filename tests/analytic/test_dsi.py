import pytest

from dsi.analytic.dsi import RunDSI
from dsi.analytic.si import RunSI
from dsi.configs.config_run import ConfigRunDSI
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
@pytest.mark.parametrize("k", [0, 1, 10, 100, 1000, 100000000])
def test_dsi_faster_than_si(c: float, failure_cost: float, a: float, k: int):
    try:
        config = ConfigRunDSI(c=c, failure_cost=failure_cost, a=a, k=k)
    except (NumOfTargetServersInsufficientError, DrafterSlowerThanTargetError):
        return
    si = RunSI(config)
    si_res: Result = si.run()
    dsi = RunDSI(config)
    res: Result = dsi.run()
    for cost_si, cost_dsi in zip(si_res.cost_per_run, res.cost_per_run):
        assert (
            cost_dsi <= cost_si
        ), f"DSI is never slower than SI. DSI: {cost_dsi}, SI: {cost_si}"
        assert config.S * c < cost_dsi <= config.S * failure_cost
