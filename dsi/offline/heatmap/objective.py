import numpy as np
import pandas as pd

from dsi.configs.simul.algo import ConfigDSI
from dsi.offline.simul.dsi import RunDSI
from dsi.offline.simul.si import RunSI
from dsi.types.df_heatmap import DataFrameHeatmap
from dsi.types.name import HeatmapColumn
from dsi.types.result import Result

enrichments: dict[str, callable] = {
    HeatmapColumn.speedup_dsi_vs_si: lambda df: df[HeatmapColumn.cost_si]
    / df[HeatmapColumn.cost_dsi],
    HeatmapColumn.speedup_dsi_vs_nonsi: lambda df: df[HeatmapColumn.cost_nonsi]
    / df[HeatmapColumn.cost_dsi],
    HeatmapColumn.speedup_si_vs_nonsi: lambda df: df[HeatmapColumn.cost_nonsi]
    / df[HeatmapColumn.cost_si],
}


def get_all_latencies(
    c: float, a: float, k: int, num_target_servers: None | int
) -> dict[str, float]:
    """
    Executes all the experiments, analyzes their results, and returns the results.
    """
    config = ConfigDSI(
        c=c,
        a=a,
        k=k,
        num_target_servers=num_target_servers,
    )
    si = RunSI(config)
    dsi = RunDSI(config)
    res_si: Result = si.run()
    res_dsi: Result = dsi.run()
    cost_si: float = np.array(res_si.cost_per_run).mean()
    cost_dsi: float = np.array(res_dsi.cost_per_run).mean()
    cost_nonsi: float = config.failure_cost * config.S
    return {
        HeatmapColumn.cost_si: cost_si,
        HeatmapColumn.cost_nonsi: cost_nonsi,
        HeatmapColumn.cost_dsi: cost_dsi,
    }


def enrich_inplace(df: pd.DataFrame) -> DataFrameHeatmap:
    """Enrich the dataframe with new columns, in-place."""
    for col, func in enrichments.items():
        df[col] = func(df)
    return DataFrameHeatmap.from_dataframe(df)
