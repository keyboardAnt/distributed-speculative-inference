import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure

from dsi.configs.plot.heatmap import ConfigPlotHeatmap
from dsi.plot.utils import savefig
from dsi.types.df_heatmap import DataFrameHeatmap
from dsi.types.name import HeatmapColumn, Param

log = logging.getLogger(__name__)


cols_to_print: dict[str, str] = {
    Param.c: "Drafter Latency",
    Param.a: "Acceptance Rate",
    Param.k: "Lookahead",
    HeatmapColumn.speedup_dsi_vs_si: "DSI Speedup over SI (x)",
    HeatmapColumn.speedup_dsi_vs_nonsi: "DSI Speedup over non-SI (x)",
    HeatmapColumn.min_speedup_dsi_vs_si: "DSI Speedup over SI (x)",
    HeatmapColumn.min_speedup_dsi_vs_nonsi: "DSI Speedup over non-SI (x)",
    HeatmapColumn.min_speedup_si_vs_nonsi: "SI Speedup over non-SI (x)",
}


class PlotHeatmap:
    """
    A class to plot a heatmap from a given DataFrameHeatmap.

    This class is initialized with a DataFrameHeatmap and provides a method to plot
    a heatmap using specific configurations provided by ConfigPlotHeatmap.
    """

    def __init__(self, df_heatmap: DataFrameHeatmap) -> None:
        self._df: DataFrameHeatmap = df_heatmap

    def plot(self, config: ConfigPlotHeatmap) -> str:
        """
        Plots the heatmap using the given configuration, saves the figure to a file, and
        returns the filepath of the saved figure.

        The heatmap's x-axis and y-axis correspond to 'drafter latency' and 'acceptance
        rate', respectively, defined in the class-level dictionary `cols_to_print`. The
        color gradient is determined by the values in `config.val_col` (e.g., speedup
        metrics), with special coloring for values below a threshold.

        Args:
            config (ConfigPlotHeatmap): The configuration for plotting, including value
            column, color scale maximum (vmax), and steps for color levels.

        Returns:
            str: The filepath where the figure is saved. This file contains the heatmap
            as configured.
        """
        vmax: float = config.vmax or self._df[config.val_col].max()
        # if vmax < 5:
        #     levels_step = .5
        levels = np.arange(0, vmax + config.levels_step, config.levels_step)

        # Setup color mapping
        colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))  # Generate colors
        pink_color = np.array([248 / 256, 24 / 256, 148 / 256, 1])  # Define pink

        # Find index for values < 1, ensuring precise application
        pink_index = np.searchsorted(levels, 1, side=config.pink_idx_side)
        # assert pink_index == 1 / levels_step, "Pink index not precise enough"
        colors[:pink_index] = pink_color

        cmap = ListedColormap(colors)
        norm = Normalize(vmin=0, vmax=vmax)

        # Create plot
        fig: Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=config.figsize)
        contour = ax.tricontourf(
            self._df[Param.c],
            self._df[Param.a],
            self._df[config.val_col],
            levels=levels,
            cmap=cmap,
            norm=norm,
        )
        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_ticks(levels)
        cbar.set_ticklabels(
            [f"{x:.0f}" if x % 1 == 0 else f"{x:.1f}" for x in levels[:-1]]
            + [f">{levels[-1]:.0f}" if levels[-1] % 1 == 0 else f">{levels[-1]:.1f}"]
        )
        # if there are more than 20 ticks labels
        if len(levels) > 20:
            # reduce the size of the ticks
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(20))

        xlabel: str = cols_to_print.get(Param.c, Param.c)
        ylabel: str = cols_to_print.get(Param.a, Param.a)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title: str = f"{config.val_col}"
        filepath: str = savefig(fig=fig, name=title)
        return filepath
