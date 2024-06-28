from dsi.analytic.heatmap.objective import get_all_latencies
from dsi.types.result import HeatmapColumn


def test_get_all_latencies():
    c = 0.5
    a = 0.8
    k = 10
    result = get_all_latencies(c, a, k)
    assert isinstance(result, dict)
    assert HeatmapColumn.cost_si in result
    assert HeatmapColumn.cost_nonsi in result
    assert HeatmapColumn.cost_dsi in result
    assert isinstance(result[HeatmapColumn.cost_si], float)
    assert isinstance(result[HeatmapColumn.cost_nonsi], float)
    assert isinstance(result[HeatmapColumn.cost_dsi], float)
