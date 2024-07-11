from dsi.configs.plt.heatmap import ConfigVisHeatmap
from dsi.vis.heatmap import ones_fn

config_vis: list[ConfigVisHeatmap] = [
    ConfigVisHeatmap(
        col_speedup="min_speedup_fed_vs_baseline",
        mask_fn=ones_fn,
        levels_step=0.1,
        vmax=1.6,
        pink_idx_side="right",
    ),
    ConfigVisHeatmap(
        col_speedup="min_speedup_fed_vs_spec",
        mask_fn=ones_fn,
        levels_step=0.2,
        vmax=2,
    ),
    ConfigVisHeatmap(
        col_speedup="min_speedup_spec_vs_nonspec",
        mask_fn=ones_fn,
        levels_step=1,
        vmax=10,
    ),
    ConfigVisHeatmap(
        col_speedup="min_speedup_fed_vs_nonspec",
        mask_fn=ones_fn,
        levels_step=0.1,
        vmax=1.4,
        pink_idx_side="right",
    ),
]
