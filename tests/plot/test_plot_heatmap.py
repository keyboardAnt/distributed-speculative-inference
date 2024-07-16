from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from dsi.plot.heatmap import ConfigPlotHeatmap, DataFrameHeatmap, PlotHeatmap
from dsi.types.name import HeatmapColumn, Param


@pytest.fixture
def heatmap_data():
    # Prepare and return test data for the heatmap
    data = pd.DataFrame(
        {
            Param.a.value: [1, 2, 2, 3],
            Param.c.value: [0.1, 0.1, 0.2, 0.3],
            HeatmapColumn.speedup_dsi_vs_si.value: [
                1.5,
                2.5,
                2.0,
                3.0,
            ],  # No NaN values here
        }
    )
    return DataFrameHeatmap(data)


@pytest.fixture
def plot_config():
    # Ensure vmax is set properly and not leading to NaN
    return ConfigPlotHeatmap(
        val_col=HeatmapColumn.speedup_dsi_vs_si,
        figsize=(10, 8),
        vmax=3.0,  # Explicit non-NaN value
        levels_step=0.5,
    )


def test_plot(mocker, heatmap_data, plot_config):
    mocker.patch("dsi.plot.heatmap.savefig", return_value="path/to/figure.png")
    mock_safe_arange = mocker.patch(
        "dsi.plot.heatmap.safe_arange", return_value=np.array([1, 2, 3])
    )
    mocker.patch("matplotlib.pyplot.subplots", return_value=(MagicMock(), MagicMock()))

    plotter = PlotHeatmap(heatmap_data)
    # Execute the plot method while capturing logs if needed
    result_path = plotter.plot(plot_config)

    mock_safe_arange.assert_called()
    assert result_path == "path/to/figure.png"
