import numpy as np
import pandas as pd

from dsi.analytic.dsi import RunDSI
from dsi.analytic.si import RunSI
from dsi.configs.config_run import ConfigRunDSI
from dsi.types.result import HeatmapColumn, Result

enrichments: dict[str, callable] = {
    HeatmapColumn.speedup_dsi_vs_si: lambda df: df[HeatmapColumn.cost_si]
    / df[HeatmapColumn.cost_dsi],
    HeatmapColumn.speedup_dsi_vs_nonsi: lambda df: df[HeatmapColumn.cost_nonsi]
    / df[HeatmapColumn.cost_dsi],
    HeatmapColumn.speedup_si_vs_nonsi: lambda df: df[HeatmapColumn.cost_nonsi]
    / df[HeatmapColumn.cost_si],
}


def get_all_latencies(c: float, a: float, k: int) -> dict[str, float]:
    """
    Executes all the experiments, analyzes their results, and returns the results.
    """
    config = ConfigRunDSI(
        a=a,
        c=c,
        k=k,
    )
    run_si = RunSI(config)
    res_si: Result = run_si.run()
    cost_si: float = np.array(res_si.cost_per_run).mean()
    # TODO(Nadav): Remove the following commented-out code.
    # try:
    #     run_dsi: RunDSI = RunDSI(config=config_dsi)
    #     res_dsi: Result = run_dsi.run()
    #     cost_dsi: float = np.array(res_dsi.cost_per_run).mean()
    # except NumOfTargetServersInsufficientError:
    #     cost_dsi: float = np.nan
    run_dsi = RunDSI(config)
    res_dsi: Result = run_dsi.run()
    cost_dsi: float = np.array(res_dsi.cost_per_run).mean()
    cost_nonsi: float = config.failure_cost * config.S
    # speedup_dsi_vs_si: float = cost_si / cost_dsi
    # speedup_dsi_vs_nonsi: float = cost_nonsi / cost_dsi
    # speedup_si_vs_nonsi: float = cost_nonsi / cost_si
    return {
        HeatmapColumn.cost_si: cost_si,
        HeatmapColumn.cost_nonsi: cost_nonsi,
        HeatmapColumn.cost_dsi: cost_dsi,
        # Objective.speedup_dsi_vs_si: speedup_dsi_vs_si,
        # Objective.speedup_dsi_vs_nonsi: speedup_dsi_vs_nonsi,
        # Objective.speedup_si_vs_nonsi: speedup_si_vs_nonsi,
    }


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich the dataframe with new columns."""
    for col, func in enrichments.items():
        df[col] = func(df)
    return df
