from pydantic import BaseModel

from dsi.configs.plot.heatmap import ConfigPlotHeatmap
from dsi.types.name import HeatmapColumn
from dsi.types.plot import PinkIndexSide


class Plots(BaseModel):
    heatmaps: list[ConfigPlotHeatmap] = [
        ConfigPlotHeatmap(
            val_col=HeatmapColumn.min_speedup_dsi_vs_baseline,
            levels_step=0.1,
            vmax=1.6,
            pink_idx_side=PinkIndexSide.right,
        ),
        ConfigPlotHeatmap(
            val_col=HeatmapColumn.min_speedup_dsi_vs_si,
            levels_step=0.2,
            vmax=2,
        ),
        ConfigPlotHeatmap(
            val_col=HeatmapColumn.min_speedup_si_vs_nonsi,
            levels_step=1,
            vmax=10,
        ),
        ConfigPlotHeatmap(
            val_col=HeatmapColumn.min_speedup_dsi_vs_nonsi,
            levels_step=0.1,
            vmax=1.4,
            pink_idx_side=PinkIndexSide.right,
        ),
    ]
