import pytest

from dsi.analytic.si import RunSI
from dsi.types.config_run import ConfigRun
from dsi.types.result import ResultSI


def test_si_result_shapes():
    config = ConfigRun(num_repeats=7)
    si = RunSI(config)
    res: ResultSI = si.run()
    assert len(res.cost_per_run) == 7
    assert len(res.num_iters_per_run) == 7


@pytest.mark.parametrize("a", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
def test_si_result_cost(a: float):
    config = ConfigRun(a=1)
    si = RunSI(config)
    res: ResultSI = si.run()
    cost_per_iter: float = config.k * config.c + config.failure_cost
    for cost, num_iters in zip(res.cost_per_run, res.num_iters_per_run):
        assert cost == cost_per_iter * num_iters
