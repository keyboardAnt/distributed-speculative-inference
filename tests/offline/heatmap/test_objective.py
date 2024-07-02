import numpy as np
import pandas as pd
import pytest

from dsi.configs.run import ConfigRunDSI
from dsi.offline.heatmap.objective import enrich_inplace, get_all_latencies
from dsi.types.name import HeatmapColumn


@pytest.mark.parametrize("c", [0.01, 0.1, 0.5, 0.8, 0.99])
@pytest.mark.parametrize("a", [0.01, 0.1, 0.5, 0.8, 0.99])
@pytest.mark.parametrize("k", [1, 2, 5, 20, 99997, 100003])
def test_get_all_latencies(c: float, a: float, k: int):
    result: dict[str, float] = get_all_latencies(c, a, k, num_target_servers=None)
    assert isinstance(result, dict)
    assert HeatmapColumn.cost_si in result
    assert HeatmapColumn.cost_nonsi in result
    assert HeatmapColumn.cost_dsi in result
    assert isinstance(result[HeatmapColumn.cost_si], float)
    assert isinstance(result[HeatmapColumn.cost_nonsi], float)
    assert isinstance(result[HeatmapColumn.cost_dsi], float)
    config = ConfigRunDSI(c=c, a=a, k=k, num_target_servers=None)
    print("Testing DSI's cost")
    num_iterations_min: int = config.S // (config.k + 1)
    dsi_cost_min: float = (num_iterations_min - 1) * config.c + config.failure_cost
    dsi_cost_max: float = config.S * (config.c * config.k + config.failure_cost)
    assert dsi_cost_min <= result[HeatmapColumn.cost_dsi] or np.isclose(
        dsi_cost_min, result[HeatmapColumn.cost_dsi]
    )
    assert result[HeatmapColumn.cost_dsi] <= dsi_cost_max or np.isclose(
        dsi_cost_max, result[HeatmapColumn.cost_dsi]
    )
    print("Testing SI's cost")
    assert (
        (config.S // (config.k + 1)) * (config.failure_cost + config.c * config.k)
        <= result[HeatmapColumn.cost_si]
        <= config.S * (config.failure_cost + config.c * config.k)
    )
    assert result[HeatmapColumn.cost_nonsi] == config.S * config.failure_cost
    assert result[HeatmapColumn.cost_dsi] <= result[
        HeatmapColumn.cost_si
    ] or np.isclose(result[HeatmapColumn.cost_si], result[HeatmapColumn.cost_dsi])


@pytest.fixture
def sample_dataframe():
    data = {
        HeatmapColumn.cost_si: [2, 4, 6],
        HeatmapColumn.cost_dsi: [1, 2, 3],
        HeatmapColumn.cost_nonsi: [3, 6, 9],
    }
    return pd.DataFrame(data)


def test_enrich_inplace_adds_columns(sample_dataframe):
    enriched_df = enrich_inplace(sample_dataframe)
    assert HeatmapColumn.speedup_dsi_vs_si in enriched_df.columns
    assert HeatmapColumn.speedup_dsi_vs_nonsi in enriched_df.columns
    assert HeatmapColumn.speedup_si_vs_nonsi in enriched_df.columns


def test_enrich_inplace_correct_calculations(sample_dataframe):
    enriched_df = enrich_inplace(sample_dataframe)
    expected_speedup_dsi_vs_si = (
        sample_dataframe[HeatmapColumn.cost_si]
        / sample_dataframe[HeatmapColumn.cost_dsi]
    )
    expected_speedup_dsi_vs_si.name = HeatmapColumn.speedup_dsi_vs_si
    expected_speedup_dsi_vs_nonsi = (
        sample_dataframe[HeatmapColumn.cost_nonsi]
        / sample_dataframe[HeatmapColumn.cost_dsi]
    )
    expected_speedup_dsi_vs_nonsi.name = HeatmapColumn.speedup_dsi_vs_nonsi
    expected_speedup_si_vs_nonsi = (
        sample_dataframe[HeatmapColumn.cost_nonsi]
        / sample_dataframe[HeatmapColumn.cost_si]
    )
    expected_speedup_si_vs_nonsi.name = HeatmapColumn.speedup_si_vs_nonsi
    pd.testing.assert_series_equal(
        enriched_df[HeatmapColumn.speedup_dsi_vs_si], expected_speedup_dsi_vs_si
    )
    pd.testing.assert_series_equal(
        enriched_df[HeatmapColumn.speedup_dsi_vs_nonsi], expected_speedup_dsi_vs_nonsi
    )
    pd.testing.assert_series_equal(
        enriched_df[HeatmapColumn.speedup_si_vs_nonsi], expected_speedup_si_vs_nonsi
    )


def test_enrich_inplace_does_not_alter_existing_columns(sample_dataframe):
    original_columns = sample_dataframe.columns.tolist()
    enriched_df = enrich_inplace(sample_dataframe)
    for column in original_columns:
        pd.testing.assert_series_equal(
            sample_dataframe[column], enriched_df[column], check_names=False
        )
