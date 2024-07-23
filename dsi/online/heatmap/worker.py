import numpy as np

from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.configs.experiment.simul.online import ConfigDSIOnline, SimulType
from dsi.online.simul.simul import SimulOnline
from dsi.types.heatmap.worker import _Worker
from dsi.types.name import HeatmapColumn
from dsi.types.result import ResultSimul, ResultWorker


class WorkerOnline(_Worker):
    def _run(self, config: ConfigDSI) -> ResultWorker:
        """
        Executes all the simulations and averages the results over their repeats.
        """
        cfg_dsi = ConfigDSIOnline.from_offline(config)
        cfg_si = ConfigDSIOnline.from_offline(config)
        cfg_si.simul_type = SimulType.SI
        dsi = SimulOnline(cfg_dsi)
        si = SimulOnline(cfg_si)
        res_si: ResultSimul = si.run()
        res_dsi: ResultSimul = dsi.run()
        cost_si: float = np.array(res_si.cost_per_repeat).mean()
        cost_dsi: float = np.array(res_dsi.cost_per_repeat).mean()
        cost_nonsi: float = config.failure_cost * config.S
        return ResultWorker(
            **{
                HeatmapColumn.cost_si: cost_si,
                HeatmapColumn.cost_nonsi: cost_nonsi,
                HeatmapColumn.cost_dsi: cost_dsi,
            }
        )
