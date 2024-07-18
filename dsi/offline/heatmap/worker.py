import numpy as np

from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.offline.simul.dsi import SimulDSI
from dsi.offline.simul.si import SimulSI
from dsi.types.heatmap.worker import _Worker
from dsi.types.name import HeatmapColumn
from dsi.types.result import ResultSimul


class Worker(_Worker):
    def _run(self, config: ConfigDSI) -> tuple[int, dict[str, float]]:
        """
        Executes all the simulations and averages the results over their repeats.
        """
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
