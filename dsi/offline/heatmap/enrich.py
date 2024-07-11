import numpy as np
import pandas as pd

from dsi.types.df_heatmap import DataFrameHeatmap
from dsi.types.name import HeatmapColumn, Param


def get_enriched_min_speedups(df: pd.DataFrame) -> pd.DataFrame:
    # Set c and a as indices
    df = df.set_index([Param.c, Param.a])

    # Cost of the baseline
    # --- 1. Original code ---
    # df["cost_baseline"] = df.apply(
    #     lambda row: min(row["cost_spec"], row["cost_nonspec"]), axis=1
    # )
    # --- 2. Efficient calculation of min ---
    # df["cost_baseline"] = np.minimum(df["cost_spec"], df["cost_nonspec"])
    # --- 3. Replacing the hardcoded column names ---
    df[HeatmapColumn.cost_baseline] = np.minimum(
        df[HeatmapColumn.cost_si], df[HeatmapColumn.cost_nonsi]
    )

    # Calculate the minimum costs for each group
    min_cost_fed: pd.Series = df.groupby(level=[Param.c, Param.a])[
        HeatmapColumn.cost_dsi
    ].transform("min")
    min_cost_spec: pd.Series = df.groupby(level=[Param.c, Param.a])[
        HeatmapColumn.cost_si
    ].transform("min")
    min_cost_baseline: pd.Series = df.groupby(level=[Param.c, Param.a])[
        HeatmapColumn.cost_baseline
    ].transform("min")

    # Calculate speedups using the correctly aligned min values
    df[HeatmapColumn.min_speedup_dsi_vs_si] = min_cost_spec / min_cost_fed
    df[HeatmapColumn.min_speedup_dsi_vs_nonsi] = (
        df[HeatmapColumn.cost_nonsi] / min_cost_fed
    )
    df[HeatmapColumn.min_speedup_si_vs_nonsi] = (
        df[HeatmapColumn.cost_nonsi] / min_cost_spec
    )
    df[HeatmapColumn.min_cost_baseline] = min_cost_baseline
    df[HeatmapColumn.min_speedup_dsi_vs_baseline] = min_cost_baseline / min_cost_fed
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
