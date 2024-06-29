import numpy as np
import pandas as pd

from dsi.analytic.dsi import RunDSI
from dsi.analytic.si import RunSI
from dsi.configs.config_run import ConfigRunDSI
from dsi.types.result import HeatmapColumn, Result
from dsi.utils import set_random_seed

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
    set_random_seed()
    res_si: Result = si.run()
    set_random_seed()
    res_dsi: Result = dsi.run()
    cost_si: float = np.array(res_si.cost_per_run).mean()
    cost_dsi: float = np.array(res_dsi.cost_per_run).mean()
    cost_nonsi: float = config.failure_cost * config.S
    return {
        HeatmapColumn.cost_si: cost_si,
        HeatmapColumn.cost_nonsi: cost_nonsi,
        HeatmapColumn.cost_dsi: cost_dsi,
    }


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the dataframe with new columns."""
    for col, func in enrichments.items():
        df[col] = func(df)
    return df
