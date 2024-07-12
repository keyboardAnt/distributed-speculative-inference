import pandas as pd
import pytest

from dsi.configs.plot.heatmap import (
    ConfigPlotHeatmap,
    ConfigPlotHeatmapInvalidLevelsStepError,
)
from dsi.types.name import HeatmapColumn
from dsi.types.plot import PinkIndexSide


def test_config_plot_heatmap_valid_levels_step():
    # Test case for a valid levels_step value
    config = ConfigPlotHeatmap(
        col_speedup=HeatmapColumn.min_speedup_dsi_vs_si,
        mask_fn=lambda df: pd.Series(True, index=df.index),
        levels_step=0.5,
        vmax=None,
        pink_idx_side=PinkIndexSide.left,
    )
    assert config.levels_step == 0.5


def test_config_plot_heatmap_invalid_levels_step():
    # Test case for an invalid levels_step value
    with pytest.raises(ConfigPlotHeatmapInvalidLevelsStepError):
        ConfigPlotHeatmap(
            col_speedup=HeatmapColumn.min_speedup_dsi_vs_si,
            mask_fn=lambda df: pd.Series(True, index=df.index),
            levels_step=0.3,
            vmax=None,
            pink_idx_side=PinkIndexSide.left,
        )


def test_config_plot_heatmap_default_values():
    # Test case for default values
    config = ConfigPlotHeatmap(
        col_speedup=HeatmapColumn.min_speedup_dsi_vs_si,
        mask_fn=lambda df: pd.Series(True, index=df.index),
    )
    assert config.levels_step == 1.0
    assert config.vmax is None
    assert config.pink_idx_side == PinkIndexSide.left
