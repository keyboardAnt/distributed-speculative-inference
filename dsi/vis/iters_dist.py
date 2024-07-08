import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from dsi.vis.cost_dist import PlotCost


class PlotIters(PlotCost):
    def _get_axes(self) -> tuple[Figure, list[plt.Axes]]:
        return plt.subplots(1, 2, figsize=(10, 5))

    def plot(self) -> None:
        super().plot()
        self._plot_num_iters(self._axes[1])

    def _plot_num_iters(self, ax: plt.Axes) -> None:
        ax.hist(
            self.result.num_iters_per_repeat,
            bins=30,
            density=True,
            color="blue",
            alpha=0.7,
        )
        self._plot_mean(ax=ax, val=np.mean(self.result.num_iters_per_repeat))
        ax.set_xlabel("Number of Iterations (i)")
        ax.set_ylabel("Frequency")
        ax.set_title("Iterations Distribution")
