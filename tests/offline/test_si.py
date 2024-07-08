import numpy as np
import pytest

from dsi.configs.run.algo import ConfigSI
from dsi.offline.run.si import RunSI
from dsi.types.result import Result


def test_si_result_shapes():
    config = ConfigSI(num_repeats=7)
    si = RunSI(config)
    res: Result = si.run()
    assert len(res.cost_per_run) == 7
    assert len(res.num_iters_per_run) == 7


@pytest.mark.parametrize("a", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
@pytest.mark.parametrize("k", [1, 3, 7, 10, 1000])
@pytest.mark.parametrize("S", [1, 10, 100, 1000])
def test_si_result_cost(a: float, k: int, S: int):
    config = ConfigSI(a=1, k=k, S=S)
    si = RunSI(config)
    res: Result = si.run()
    cost_per_iter_max: float = config.k * config.c + config.failure_cost
    cost_per_iter_min: float = config.failure_cost
    for cost, num_iters in zip(res.cost_per_run, res.num_iters_per_run):
        cost_min: float = cost_per_iter_min * num_iters
        cost_max: float = cost_per_iter_max * num_iters
        assert cost_min <= cost or np.isclose(cost_min, cost)
        assert cost <= cost_max or np.isclose(cost_max, cost)
