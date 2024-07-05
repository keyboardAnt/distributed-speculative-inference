import os
from datetime import datetime
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.colors import ListedColormap, Normalize
from pydantic import BaseModel

# class VisHeatmap:
#     def __init__(self, df: DataFrameHeatmap) -> None:
#         self._df: DataFrameHeatmap = df

#     def plot(self, config: ConfigVisHeatmap):
#         raise NotImplementedError


cols_to_print: dict[str, str] = {
    "c": "Drafter Latency",
    "a": "Acceptance Rate",
    "k": "Lookahead",
    "speedup_fed_vs_spec": "DSI Speedup over SI (x)",
    "speedup_fed_vs_nonspec": "DSI Speedup over non-SI (x)",
    "min_speedup_fed_vs_spec": "DSI Speedup over SI (x)",
    "min_speedup_fed_vs_nonspec": "DSI Speedup over non-SI (x)",
    "min_speedup_spec_vs_nonspec": "SI Speedup over non-SI (x)",
}


def _plot_contour(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    val_col: str,
    levels_step: float = 1.0,
    vmax: float | None = None,
    pink_idx_side: Literal["left", "right"] = "left",
) -> plt.Figure:
    assert levels_step <= 1, "Levels step must be less than or equal to 1"
    assert ((1 / levels_step) % 1) == 0, "Levels step must be a factor of 1"
    vmax: float = vmax or df[val_col].max()
    # if vmax < 5:
    #     levels_step = .5
    levels = np.arange(0, vmax + levels_step, levels_step)

    # Setup color mapping
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))  # Generate colors
    pink_color = np.array([248 / 256, 24 / 256, 148 / 256, 1])  # Define pink

    # Find index for values < 1, ensuring precise application
    pink_index = np.searchsorted(levels, 1, side=pink_idx_side)
    # assert pink_index == 1 / levels_step, "Pink index not precise enough"
    colors[:pink_index] = pink_color

    cmap = ListedColormap(colors)
    norm = Normalize(vmin=0, vmax=vmax)

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.tricontourf(
        df[x_col], df[y_col], df[val_col], levels=levels, cmap=cmap, norm=norm
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

    xlabel: str = cols_to_print.get(x_col, x_col)
    ylabel: str = cols_to_print.get(y_col, y_col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return fig


def _get_enriched_min_speedups(df: pd.DataFrame) -> pd.DataFrame:
    # Set c and a as indices
    df = df.set_index(["c", "a"])

    # Cost of the baseline
    df["cost_baseline"] = df.apply(
        lambda row: min(row["cost_spec"], row["cost_nonspec"]), axis=1
    )

    # Calculate the minimum costs for each group
    min_cost_fed: pd.Series = df.groupby(level=["c", "a"])["cost_fed"].transform("min")
    min_cost_spec: pd.Series = df.groupby(level=["c", "a"])["cost_spec"].transform(
        "min"
    )
    min_cost_baseline: pd.Series = df.groupby(level=["c", "a"])[
        "cost_baseline"
    ].transform("min")

    # Calculate speedups using the correctly aligned min values
    df["min_speedup_fed_vs_spec"] = min_cost_spec / min_cost_fed
    df["min_speedup_fed_vs_nonspec"] = df["cost_nonspec"] / min_cost_fed
    df["min_speedup_spec_vs_nonspec"] = df["cost_nonspec"] / min_cost_spec
    df["min_cost_baseline"] = min_cost_baseline
    df["min_speedup_fed_vs_baseline"] = min_cost_baseline / min_cost_fed
    return df.reset_index()


csv_filepath = "results/offline/heatmap/heatmap-20240702-012750.csv"
df: pd.DataFrame = pd.read_csv(csv_filepath, index_col=0)
mask_ones: np.ndarray = np.ones_like(df.index, dtype=bool)


_plot_contour(
    _get_enriched_min_speedups(df[mask_ones]),
    "c",
    "a",
    "min_speedup_fed_vs_spec",
)


def ones_fn(df: pd.DataFrame) -> pd.Series:
    return np.ones_like(df.index, dtype=bool)


col_to_masks = {
    "min_speedup_fed_vs_spec": [ones_fn],
    "min_speedup_spec_vs_nonspec": [ones_fn],
    # "min_speedup_fed_vs_nonspec": [get_mask_fast_plot, get_mask_slow_plot],
    "min_speedup_fed_vs_nonspec": [ones_fn],
}


class PlotSpeedupConfig(BaseModel):
    col_speedup: str
    mask_fn: Callable[[pd.DataFrame], pd.Series]
    levels_step: None | float = None
    vmax: None | float = None
    pink_idx_side: Literal["left", "right"] = "left"


configs: list[PlotSpeedupConfig] = [
    PlotSpeedupConfig(
        col_speedup="min_speedup_fed_vs_baseline",
        mask_fn=ones_fn,
        levels_step=0.1,
        vmax=1.6,
        pink_idx_side="right",
    ),
    PlotSpeedupConfig(
        col_speedup="min_speedup_fed_vs_spec",
        mask_fn=ones_fn,
        levels_step=0.2,
        vmax=2,
    ),
    PlotSpeedupConfig(
        col_speedup="min_speedup_spec_vs_nonspec",
        mask_fn=ones_fn,
        levels_step=1,
        vmax=10,
    ),
    PlotSpeedupConfig(
        col_speedup="min_speedup_fed_vs_nonspec",
        mask_fn=ones_fn,
        levels_step=0.1,
        vmax=1.4,
        pink_idx_side="right",
    ),
]


now = datetime.now()
dirpath = (
    f"/Users/nadavt/repos/federated-inference/outs/{now.strftime('%Y%m%d-%H%M%S')}"
)


def plot_speedup(config: PlotSpeedupConfig) -> None:
    fig: plt.Figure = _plot_contour(
        df=_get_enriched_min_speedups(df[config.mask_fn(df)]),
        x_col="c",
        y_col="a",
        val_col=config.col_speedup,
        levels_step=config.levels_step,
        vmax=config.vmax,
        pink_idx_side=config.pink_idx_side,
    )
    title: str = f"{config.col_speedup} - {config.mask_fn.__name__}"
    # fig.suptitle(title)
    filepath: str = os.path.join(dirpath, title)
    plt.tight_layout()
    fig.savefig(filepath + ".pdf", dpi=300, format="pdf", bbox_inches="tight")


for config in configs:
    plot_speedup(config)
