from dsi.analytic.dsi import RunDSI
from dsi.types.config_run import ConfigRunDSI
from dsi.types.results import Result


def test_dsi_result_shapes():
    num_repeats: int = 17
    config = ConfigRunDSI(num_repeats=num_repeats)
    dsi = RunDSI(config)
    res: Result = dsi.run()
    assert (
        len(res.cost_per_run) == num_repeats
    ), f"expected {num_repeats} results, got {len(res.cost_per_run)}"
