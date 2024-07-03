import numpy as np
import pandas as pd

from dsi.configs.config_run import ConfigRunDSI
from dsi.offline.run.dsi import RunDSI
from dsi.offline.run.si import RunSI
from dsi.types.result import HeatmapColumn, Result

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
    config = ConfigRunDSI(
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


def enrich_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the dataframe with new columns, in-place."""
    for col, func in enrichments.items():
        df[col] = func(df)
    return df
