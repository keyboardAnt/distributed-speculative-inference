import pytest

from dsi.configs.plot.heatmap import (
    ConfigPlotHeatmap,
    ConfigPlotHeatmapInvalidLevelsStepError,
)
from dsi.types.name import HeatmapColumn


def test_config_plot_heatmap_valid_levels_step():
    # Test case for a valid levels_step value
    config = ConfigPlotHeatmap(
        val_col=HeatmapColumn.min_speedup_dsi_vs_si,
        levels_step=0.5,
        vmax=None,
    )
    assert config.levels_step == 0.5


def test_config_plot_heatmap_invalid_levels_step():
    # Test case for an invalid levels_step value
    with pytest.raises(ConfigPlotHeatmapInvalidLevelsStepError):
        ConfigPlotHeatmap(
            val_col=HeatmapColumn.min_speedup_dsi_vs_si,
            levels_step=0.3,
            vmax=None,
        )


def test_config_plot_heatmap_default_values():
    # Test case for default values
    config = ConfigPlotHeatmap(
        val_col=HeatmapColumn.min_speedup_dsi_vs_si,
    )
    assert config.levels_step == 1.0
    assert config.vmax is None
