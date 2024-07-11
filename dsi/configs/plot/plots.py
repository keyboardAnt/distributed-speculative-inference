from dsi.configs.plot.heatmap import ConfigPlotHeatmap
from dsi.plot.heatmap import ones_fn

config_plot_heatmaps: list[ConfigPlotHeatmap] = [
    ConfigPlotHeatmap(
        col_speedup="min_speedup_fed_vs_baseline",
        mask_fn=ones_fn,
        levels_step=0.1,
        vmax=1.6,
        pink_idx_side="right",
    ),
    ConfigPlotHeatmap(
        col_speedup="min_speedup_fed_vs_spec",
        mask_fn=ones_fn,
        levels_step=0.2,
        vmax=2,
    ),
    ConfigPlotHeatmap(
        col_speedup="min_speedup_spec_vs_nonspec",
        mask_fn=ones_fn,
        levels_step=1,
        vmax=10,
    ),
    ConfigPlotHeatmap(
        col_speedup="min_speedup_fed_vs_nonspec",
        mask_fn=ones_fn,
        levels_step=0.1,
        vmax=1.4,
        pink_idx_side="right",
    ),
]
