from dsi.analytic.heatmap.objective import Column, get_all_latencies


def test_get_all_latencies():
    c = 0.5
    a = 0.8
    k = 10
    result = get_all_latencies(c, a, k)
    assert isinstance(result, dict)
    assert Column.cost_si in result
    assert Column.cost_nonsi in result
    assert Column.cost_dsi in result
    assert isinstance(result[Column.cost_si], float)
    assert isinstance(result[Column.cost_nonsi], float)
    assert isinstance(result[Column.cost_dsi], float)
