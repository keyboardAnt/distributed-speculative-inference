import pytest

from dsi.analytic.heatmap.objective import get_all_latencies
from dsi.configs.config_run import ConfigRunDSI
from dsi.types.result import HeatmapColumn


@pytest.mark.parametrize("c", [0.01, 0.1, 0.5, 0.8, 0.99])
@pytest.mark.parametrize("a", [0.01, 0.1, 0.5, 0.8, 0.99])
@pytest.mark.parametrize("k", [1, 2, 5, 20, 99997, 100003])
def test_get_all_latencies(c: float, a: float, k: int):
    result: dict[str, float] = get_all_latencies(c, a, k)
    assert isinstance(result, dict)
    assert HeatmapColumn.cost_si in result
    assert HeatmapColumn.cost_nonsi in result
    assert HeatmapColumn.cost_dsi in result
    assert isinstance(result[HeatmapColumn.cost_si], float)
    assert isinstance(result[HeatmapColumn.cost_nonsi], float)
    assert isinstance(result[HeatmapColumn.cost_dsi], float)
    config = ConfigRunDSI(c=c, a=a, k=k, num_target_servers=None)
    assert (
        config.S * c < result[HeatmapColumn.cost_dsi] <= config.S * config.failure_cost
    )
    assert (
        config.S * c
        < result[HeatmapColumn.cost_si]
        <= config.S * (config.k + 1) * config.failure_cost
    )
    assert result[HeatmapColumn.cost_nonsi] == config.S * config.failure_cost
