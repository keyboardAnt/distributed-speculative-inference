import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.figure import Figure

from dsi.configs.plot.heatmap import ConfigPlotHeatmap
from dsi.plot.utils import savefig
from dsi.types.df_heatmap import DataFrameHeatmap
from dsi.types.name import HeatmapColumn, Param, Print

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
        """
        Plots the heatmap using the given configuration, saves the figure to a file, and
        returns the filepath of the saved figure.

        The heatmap's x-axis and y-axis correspond to 'drafter latency' and 'acceptance
        rate', respectively, defined in the class-level dictionary `Print`. The color
        gradient is determined by the values in `config.val_col` (e.g., speedup
        metrics), with special coloring for values below a threshold.

        Args:
            config (ConfigPlotHeatmap): The configuration for plotting, including value
            column, color scale maximum (vmax), and steps for color levels.

        Returns:
            str: The filepath where the figure is saved. This file contains the heatmap
            as configured.
        """
        # vmax: float = config.vmax or self._df[config.val_col].max()
        # # if vmax < 5:
        # #     levels_step = .5
        # levels = np.arange(0, vmax + config.levels_step, config.levels_step)

        # # Setup color mapping
        # colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))  # Generate colors
        # pink_color = np.array([248 / 256, 24 / 256, 148 / 256, 1])  # Define pink

        # # Find and apply pink mask
        # pink_mask = self._get_pink_mask(levels, 1)
        # colors[pink_mask] = pink_color

        # cmap = ListedColormap(colors)
        # norm = Normalize(vmin=0, vmax=vmax)

        cmap, norm, levels = self._initialize_color_mapping(config.val_col)

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
        name_to_print: dict[Param, str] = Print().name_to_print
        ax.set_xlabel(name_to_print[Param.c])
        ax.set_ylabel(name_to_print[Param.a])
        filepath: str = savefig(fig=fig, name=config.val_col.value)
        return filepath

    def _initialize_color_mapping(self, val_col: HeatmapColumn):
        """
        Initializes the color mapping (cmap and norm) for the heatmap.

        Args:
            val_col (HeatmapColumn): The value column for determining the color mapping.

        Returns:
            tuple: cmap (ListedColormap), norm (Normalize), levels (np.ndarray)
        """
        vmax: float = self._df[val_col.value].max()
        levels_step = 0.2
        levels = np.arange(0, vmax + levels_step, levels_step)

        # Setup color mapping
        colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))  # Generate colors
        pink_color = np.array([248 / 256, 24 / 256, 148 / 256, 1])  # Define pink

        # Find and apply pink mask
        pink_mask = self._get_pink_mask(levels, 1)
        colors[pink_mask] = pink_color

        cmap = ListedColormap(colors)
        norm = Normalize(vmin=0, vmax=vmax)

        return cmap, norm, levels

    @staticmethod
    def _get_pink_mask(levels: np.ndarray, threshold: float) -> np.ndarray:
        """
        Returns a mask for applying pink color to levels below the given threshold.

        Args:
            levels (np.ndarray): The levels array.
            threshold (float): The threshold value for applying pink color.

        Returns:
            np.ndarray: The mask array with True for values to be colored pink.
        """
        # Determine the side for np.searchsorted
        side = "right" if threshold in levels else "left"
        pink_index = np.searchsorted(levels, threshold, side=side)
        mask = np.zeros_like(levels, dtype=bool)
        mask[:pink_index] = True
        return mask
