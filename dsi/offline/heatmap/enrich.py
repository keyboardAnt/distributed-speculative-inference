import numpy as np
import pandas as pd

from dsi.types.df_heatmap import DataFrameHeatmap
from dsi.types.name import HeatmapColumn, Param

# def get_enriched_min_speedups(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.set_index([Param.c, Param.a])
#     df[HeatmapColumn.cost_baseline] = np.minimum(
#         df[HeatmapColumn.cost_si], df[HeatmapColumn.cost_nonsi]
#     )
#     # Calculate the minimum costs for each group
#     min_cost_fed: pd.Series = df.groupby(level=[Param.c, Param.a])[
#         HeatmapColumn.cost_dsi
#     ].transform("min")
#     min_cost_spec: pd.Series = df.groupby(level=[Param.c, Param.a])[
#         HeatmapColumn.cost_si
#     ].transform("min")
#     min_cost_baseline: pd.Series = df.groupby(level=[Param.c, Param.a])[
#         HeatmapColumn.cost_baseline
#     ].transform("min")

#     # Calculate speedups using the correctly aligned min values
#     df[HeatmapColumn.min_speedup_dsi_vs_si] = min_cost_spec / min_cost_fed
#     df[HeatmapColumn.min_speedup_dsi_vs_nonsi] = (
#         df[HeatmapColumn.cost_nonsi] / min_cost_fed
#     )
#     df[HeatmapColumn.min_speedup_si_vs_nonsi] = (
#         df[HeatmapColumn.cost_nonsi] / min_cost_spec
#     )
#     df[HeatmapColumn.min_cost_baseline] = min_cost_baseline
#     df[HeatmapColumn.min_speedup_dsi_vs_baseline] = min_cost_baseline / min_cost_fed
#     return df.reset_index()


def get_enriched_min_speedups(df: pd.DataFrame) -> pd.DataFrame:
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


enrichments: dict[str, callable] = {
    HeatmapColumn.speedup_dsi_vs_si: lambda df: df[HeatmapColumn.cost_si]
    / df[HeatmapColumn.cost_dsi],
    HeatmapColumn.speedup_dsi_vs_nonsi: lambda df: df[HeatmapColumn.cost_nonsi]
    / df[HeatmapColumn.cost_dsi],
    HeatmapColumn.speedup_si_vs_nonsi: lambda df: df[HeatmapColumn.cost_nonsi]
    / df[HeatmapColumn.cost_si],
}


def enrich_inplace(df: pd.DataFrame) -> DataFrameHeatmap:
    """Enrich the dataframe with new columns, in-place."""
    for col, func in enrichments.items():
        df[col] = func(df)
    return DataFrameHeatmap.from_dataframe(df)
