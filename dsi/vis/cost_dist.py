import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from dsi.types.results import Result


class PlotCost:
    def __init__(self, result: Result, suptitle: str | None = None) -> None:
        self.result: Result = result
        self._suptitle: str | None = suptitle
        fig, axes = self._get_axes()
        self._fig: plt.Figure = fig
        self._axes: list[plt.Axes] = axes

    def plot(self) -> None:
        """
        Plots the total latency distribution and the number of iterations.
        """
        if self._suptitle:
            self._fig.suptitle(self._suptitle)
        self._plot_total_costs(self._axes[0])

    def _get_axes(self) -> tuple[Figure, list[plt.Axes]]:
        fig: plt.Figure
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        return fig, [ax]

    def _plot_total_costs(self, ax: plt.Axes) -> None:
        ax.hist(
            self.result.cost_per_run,
            bins=30,
            density=True,
            color="purple",
            alpha=0.7,
        )
        self._plot_mean(ax=ax, val=np.mean(self.result.cost_per_run))
        ax.set_xlabel("Total Latency")
        ax.set_ylabel("Frequency")
        ax.set_title("Total Latency Distribution")

    def _plot_mean(self, ax: plt.Axes, val: float) -> None:
        ax.axvline(x=val, color="black", linestyle="--")
        ax.text(val, 0.02, f"Mean: {val:.2f}", color="black", fontsize=8)
