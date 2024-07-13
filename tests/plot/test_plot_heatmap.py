from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dsi.plot.heatmap import ConfigPlotHeatmap, DataFrameHeatmap, PlotHeatmap


@pytest.fixture
def sample_data():
    return {
        "c": np.random.rand(100),
        "a": np.random.rand(100),
        "min_speedup_dsi_vs_si": np.random.rand(100) * 3,
    }


@pytest.fixture
def df_heatmap(sample_data):
    return DataFrameHeatmap(sample_data)


@pytest.fixture
def config():
    return ConfigPlotHeatmap(val_col="min_speedup_dsi_vs_si", levels_step=0.2, vmax=2.5)


def test_get_pink_mask():
    levels = np.arange(0, 3, 0.5)
    threshold = 1
    expected_mask = np.array([True, True, True, False, False, False])
    assert np.array_equal(PlotHeatmap._get_pink_mask(levels, threshold), expected_mask)

    threshold = 1.5
    expected_mask = np.array([True, True, True, True, False, False])
    assert np.array_equal(PlotHeatmap._get_pink_mask(levels, threshold), expected_mask)

    threshold = 3
    expected_mask = np.array([True, True, True, True, True, True])
    assert np.array_equal(PlotHeatmap._get_pink_mask(levels, threshold), expected_mask)

    threshold = 0
    expected_mask = np.array([True, False, False, False, False, False])
    assert np.array_equal(PlotHeatmap._get_pink_mask(levels, threshold), expected_mask)


@patch("matplotlib.pyplot.subplots")
@patch(
    "hydra.core.hydra_config.HydraConfig.get",
    return_value=MagicMock(runtime=MagicMock(output_dir="mocked_output_dir")),
)
def test_plot(mock_hydra_config, mock_subplots, df_heatmap, config):
    plotter = PlotHeatmap(df_heatmap)

    fig_mock = MagicMock()
    ax_mock = MagicMock()
    mock_subplots.return_value = (fig_mock, ax_mock)

    plotter.plot(config)

    mock_subplots.assert_called_once_with(figsize=config.figsize)
    ax_mock.tricontourf.assert_called_once()
    fig_mock.colorbar.assert_called_once()
    fig_mock.savefig.assert_called_once()


@pytest.mark.parametrize(
    "threshold, expected_mask",
    [
        (1, [True, True, True, False, False, False]),
        (1.5, [True, True, True, True, False, False]),
        (3, [True, True, True, True, True, True]),
        (0, [True, False, False, False, False, False]),
    ],
)
def test_get_pink_mask_parametrized(threshold, expected_mask):
    levels = np.arange(0, 3, 0.5)
    mask = PlotHeatmap._get_pink_mask(levels, threshold)
    assert np.array_equal(mask, expected_mask)
