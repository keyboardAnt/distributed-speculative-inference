import numpy as np

from dsi.analytic.dsi import RunDSI
from dsi.analytic.si import RunSI
from dsi.types.config_run import ConfigRun, ConfigRunDSI
from dsi.types.exception import NumOfTargetServersInsufficientError
from dsi.types.result import Result


class Objective:
    cost_nonspec = "cost_nonspec"
    cost_spec = "cost_spec"
    cost_fed = "cost_fed"
    speedup_fed_vs_nonspec = "speedup_fed_vs_nonspec"
    speedup_fed_vs_spec = "speedup_fed_vs_spec"
    speedup_spec_vs_nonspec = "speedup_spec_vs_nonspec"


def calc_all(c: float, a: float, k: int) -> dict[str, float]:
    """
    Executes all the experiments, analyzes their results, and returns the results.
    """
    config_spec: ConfigRun = ConfigRun(a=a, c=c, k=k)
    config_fed: ConfigRunDSI = ConfigRunDSI(
        a=a,
        c=c,
        k=k,
    )
    run_spec: RunSI = RunSI(config=config_spec)
    res_spec: Result = run_spec.run()
    cost_spec: float = np.array(res_spec.cost_per_run).mean()
    try:
        run_fed: RunDSI = RunDSI(config=config_fed)
        res_fed: Result = run_fed.run()
        cost_fed: float = np.array(res_fed.cost_per_run).mean()
    except NumOfTargetServersInsufficientError:
        cost_fed: float = np.nan
    cost_nonspec: float = config_spec.failure_cost * config_spec.S
    speedup_fed_vs_spec: float = cost_spec / cost_fed
    speedup_fed_vs_nonspec: float = cost_nonspec / cost_fed
    speedup_spec_vs_nonspec: float = cost_nonspec / cost_spec
    return {
        Objective.cost_spec: cost_spec,
        Objective.cost_nonspec: cost_nonspec,
        Objective.cost_fed: cost_fed,
        Objective.speedup_fed_vs_spec: speedup_fed_vs_spec,
        Objective.speedup_fed_vs_nonspec: speedup_fed_vs_nonspec,
        Objective.speedup_spec_vs_nonspec: speedup_spec_vs_nonspec,
    }
