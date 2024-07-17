import numpy as np

from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.offline.simul.dsi import SimulDSI
from dsi.offline.simul.si import SimulSI
from dsi.types.name import HeatmapColumn
from dsi.types.result import ResultSimul


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
    si = SimulSI(config)
    dsi = SimulDSI(config)
    res_si: ResultSimul = si.run()
    res_dsi: ResultSimul = dsi.run()
    cost_si: float = np.array(res_si.cost_per_repeat).mean()
    cost_dsi: float = np.array(res_dsi.cost_per_repeat).mean()
    cost_nonsi: float = config.failure_cost * config.S
    return {
        HeatmapColumn.cost_si: cost_si,
        HeatmapColumn.cost_nonsi: cost_nonsi,
        HeatmapColumn.cost_dsi: cost_dsi,
    }
