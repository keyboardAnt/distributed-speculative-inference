import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import BoundaryNorm, Colormap, ListedColormap
from matplotlib.figure import Figure
from matplotlib.pyplot import get_cmap

from dsi.configs.plot.heatmap import ConfigPlotHeatmap
from dsi.plot.utils import savefig
from dsi.types.df_heatmap import DataFrameHeatmap
from dsi.types.name import Param, Print

log = logging.getLogger(__name__)


class PlotHeatmap:
    """
    A class to plot a heatmap from a given DataFrameHeatmap.

    This class is initialized with a DataFrameHeatmap and provides a method to plot
    a heatmap using specific configurations provided by ConfigPlotHeatmap.
    """

    def __init__(self, df_heatmap: DataFrameHeatmap) -> None:
        self._df: DataFrameHeatmap = df_heatmap

    def plot(self, config: ConfigPlotHeatmap) -> str:
        # Average the speedup over repeats that have the same 'drafter_latency' and
        # 'acceptance_rate' values
        df_agg = (
            self._df.groupby([Param.a, Param.c])[config.val_col].mean().reset_index()
        )
        df_pivot = df_agg.pivot(index=Param.a, columns=Param.c, values=config.val_col)
        x_unique = df_pivot.columns.values
        y_unique = df_pivot.index.values
        z_matrix = df_pivot.values
        # Setup color mapping
        vmax: float = config.vmax or z_matrix.max()
        bounds: list[float] = [0] + np.arange(
            1, vmax + config.levels_step, config.levels_step
        ).tolist()
        num_nonpink: int = len(bounds) - 1
        viridis: Colormap = get_cmap("viridis")
        colors_nonpink: np.ndarray = viridis(np.linspace(0, 1, num_nonpink))
        color_pink = np.array([248 / 256, 24 / 256, 148 / 256, 1])
        colors = np.vstack((color_pink, colors_nonpink))
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(bounds, cmap.N)
        fig: Figure
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=config.figsize)
        qmesh = ax.pcolormesh(
            x_unique,
            y_unique,
            z_matrix,
            cmap=cmap,
            norm=norm,
        )
        cbar = fig.colorbar(qmesh, ax=ax)
        cbar.set_ticks(bounds)
        # cbar.set_ticklabels(
        #     [f"{x:.0f}" if x % 1 == 0 else f"{x:.1f}" for x in levels[:-1]]
        #     + [f">{levels[-1]:.0f}" if levels[-1] % 1 == 0 else f">{levels[-1]:.1f}"]
        # )
        # if there are more than 20 ticks labels
        cbar.set_ticklabels([f"{b:.2f}" for b in bounds])
        if len(bounds) > 20:
            cbar.ax.yaxis.set_major_locator(ticker.MaxNLocator(20))
        name_to_print: dict[Param, str] = Print().name_to_print
        ax.set_xlabel(name_to_print[Param.c])
        ax.set_ylabel(name_to_print[Param.a])
        title: str = f"{config.val_col}"
        filepath: str = savefig(fig=fig, name=title)
        return filepath
