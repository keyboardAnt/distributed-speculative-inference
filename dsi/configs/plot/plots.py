from dsi.configs.plot.heatmap import ConfigPlotHeatmap
from dsi.plot.mask import mask_ones
from dsi.types.name import HeatmapColumn

config_plot_heatmaps: list[ConfigPlotHeatmap] = [
    ConfigPlotHeatmap(
        col_speedup=HeatmapColumn.min_speedup_dsi_vs_baseline,
        mask_fn=mask_ones,
        levels_step=0.1,
        vmax=1.6,
        pink_idx_side="right",
    ),
    ConfigPlotHeatmap(
        col_speedup=HeatmapColumn.min_speedup_dsi_vs_si,
        mask_fn=mask_ones,
        levels_step=0.2,
        vmax=2,
    ),
    ConfigPlotHeatmap(
        col_speedup=HeatmapColumn.min_speedup_si_vs_nonsi,
        mask_fn=mask_ones,
        levels_step=1,
        vmax=10,
    ),
    ConfigPlotHeatmap(
        col_speedup=HeatmapColumn.min_speedup_dsi_vs_nonsi,
        mask_fn=mask_ones,
        levels_step=0.1,
        vmax=1.4,
        pink_idx_side="right",
    ),
]
