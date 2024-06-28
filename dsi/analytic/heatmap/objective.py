import numpy as np

from dsi.analytic.dsi import RunDSI
from dsi.analytic.si import RunSI
from dsi.configs.config_run import ConfigRun, ConfigRunDSI
from dsi.types.result import Result


class Objective:
    cost_nonsi = "cost_nonspec"
    cost_si = "cost_spec"
    cost_dsi = "cost_fed"
    speedup_dsi_vs_nonsi = "speedup_fed_vs_nonspec"
    speedup_dsi_vs_si = "speedup_fed_vs_spec"
    speedup_si_vs_nonsi = "speedup_spec_vs_nonspec"


def calc_all(c: float, a: float, k: int) -> dict[str, float]:
    """
    Executes all the experiments, analyzes their results, and returns the results.
    """
    config_si = ConfigRun(a=a, c=c, k=k)
    config_dsi = ConfigRunDSI(
        a=a,
        c=c,
        k=k,
    )
    run_si = RunSI(config=config_si)
    res_si: Result = run_si.run()
    cost_si: float = np.array(res_si.cost_per_run).mean()
    # try:
    #     run_dsi: RunDSI = RunDSI(config=config_dsi)
    #     res_dsi: Result = run_dsi.run()
    #     cost_dsi: float = np.array(res_dsi.cost_per_run).mean()
    # except NumOfTargetServersInsufficientError:
    #     cost_dsi: float = np.nan
    run_dsi = RunDSI(config=config_dsi)
    res_dsi: Result = run_dsi.run()
    cost_dsi: float = np.array(res_dsi.cost_per_run).mean()
    cost_nonsi: float = config_si.failure_cost * config_si.S
    speedup_dsi_vs_si: float = cost_si / cost_dsi
    speedup_dsi_vs_nonsi: float = cost_nonsi / cost_dsi
    speedup_si_vs_nonsi: float = cost_nonsi / cost_si
    return {
        Objective.cost_si: cost_si,
        Objective.cost_nonsi: cost_nonsi,
        Objective.cost_dsi: cost_dsi,
        Objective.speedup_dsi_vs_si: speedup_dsi_vs_si,
        Objective.speedup_dsi_vs_nonsi: speedup_dsi_vs_nonsi,
        Objective.speedup_si_vs_nonsi: speedup_si_vs_nonsi,
    }
