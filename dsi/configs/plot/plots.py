from dsi.configs.plot.heatmap import ConfigPlotHeatmap
from dsi.plot.mask import mask_ones
from dsi.types.name import HeatmapColumn
from dsi.types.plot import PinkIndexSide

config_plot_heatmaps: list[ConfigPlotHeatmap] = [
    ConfigPlotHeatmap(
        val_col=HeatmapColumn.min_speedup_dsi_vs_baseline,
        mask_fn=mask_ones,
        levels_step=0.1,
        vmax=1.6,
        pink_idx_side=PinkIndexSide.right,
    ),
    ConfigPlotHeatmap(
        val_col=HeatmapColumn.min_speedup_dsi_vs_si,
        mask_fn=mask_ones,
        levels_step=0.2,
        vmax=2,
    ),
    ConfigPlotHeatmap(
        val_col=HeatmapColumn.min_speedup_si_vs_nonsi,
        mask_fn=mask_ones,
        levels_step=1,
        vmax=10,
    ),
    ConfigPlotHeatmap(
        val_col=HeatmapColumn.min_speedup_dsi_vs_nonsi,
        mask_fn=mask_ones,
        levels_step=0.1,
        vmax=1.4,
        pink_idx_side=PinkIndexSide.right,
    ),
]
