import numpy as np
import pandas as pd

from dsi.types.heatmap.df_heatmap import DataFrameHeatmap
from dsi.types.name import HeatmapColumn, Param


def enrich_simple_speedups(df: pd.DataFrame) -> DataFrameHeatmap:
    """Use lambda functions registered in `enrichments`."""
    enrichments: dict[str, callable] = {
        HeatmapColumn.speedup_dsi_vs_si: lambda df: df[HeatmapColumn.cost_si]
        / df[HeatmapColumn.cost_dsi],
        HeatmapColumn.speedup_dsi_vs_nonsi: lambda df: df[HeatmapColumn.cost_nonsi]
        / df[HeatmapColumn.cost_dsi],
        HeatmapColumn.speedup_si_vs_nonsi: lambda df: df[HeatmapColumn.cost_nonsi]
        / df[HeatmapColumn.cost_si],
    }
    for col, func in enrichments.items():
        df[col] = func(df)
    return DataFrameHeatmap.from_dataframe(df)


def enrich_min_speedups(df: pd.DataFrame) -> pd.DataFrame:
    # Setting index only once to be used throughout the function
    df = df.set_index([Param.c, Param.a])

    # Compute baseline costs using vectorized numpy operation
    df[HeatmapColumn.cost_baseline] = np.minimum(
        df[HeatmapColumn.cost_si], df[HeatmapColumn.cost_nonsi]
    )

    # Group once and calculate all minimums together
    grouped = df.groupby(level=[Param.c, Param.a])
    min_costs = (
        grouped[
            [HeatmapColumn.cost_dsi, HeatmapColumn.cost_si, HeatmapColumn.cost_baseline]
        ]
        .transform("min")
        .rename(
            columns={
                HeatmapColumn.cost_dsi: HeatmapColumn.min_cost_dsi,
                HeatmapColumn.cost_si: HeatmapColumn.min_cost_si,
                HeatmapColumn.cost_baseline: HeatmapColumn.min_cost_baseline,
            }
        )
    )

    # Merge calculated min values back to the original dataframe
    df = df.merge(min_costs, left_index=True, right_index=True)

    # Calculate speedups directly using vectorized operations
    df[HeatmapColumn.min_speedup_dsi_vs_si] = (
        df[HeatmapColumn.min_cost_si] / df[HeatmapColumn.min_cost_dsi]
    )
    df[HeatmapColumn.min_speedup_dsi_vs_nonsi] = (
        df[HeatmapColumn.cost_nonsi] / df[HeatmapColumn.min_cost_dsi]
    )
    df[HeatmapColumn.min_speedup_si_vs_nonsi] = (
        df[HeatmapColumn.cost_nonsi] / df[HeatmapColumn.min_cost_si]
    )
    df[HeatmapColumn.min_speedup_dsi_vs_baseline] = (
        df[HeatmapColumn.min_cost_baseline] / df[HeatmapColumn.min_cost_dsi]
    )

    return df.reset_index()


def enrich(df: pd.DataFrame) -> DataFrameHeatmap:
    """Enrich the dataframe with new columns, in-place."""
    df = enrich_simple_speedups(df)
    df = enrich_min_speedups(df)
    return DataFrameHeatmap.from_dataframe(df)
