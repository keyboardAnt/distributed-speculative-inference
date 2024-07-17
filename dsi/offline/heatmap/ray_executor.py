import logging

import numpy as np
import pandas as pd
import ray
from ray.experimental import tqdm_ray

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap
from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.offline.heatmap.params import get_df_heatmap_params
from dsi.offline.simul.dsi import SimulDSI
from dsi.offline.simul.si import SimulSI
from dsi.types.name import HeatmapColumn, Param
from dsi.types.result import ResultSimul

log = logging.getLogger(__name__)


class RayExecutor:
    def __init__(self, config: ConfigHeatmap) -> None:
        self._df_configs: pd.DataFrame = get_df_heatmap_params(config)
        self._results_raw: None | list[tuple[int, dict[str, float]]] = None
        self.df_results: pd.DataFrame = self._df_configs.copy(deep=True)

    def run(self) -> pd.DataFrame:
        # NOTE: Ray discovers and utilizes all available resources by default
        ray.init(
            ignore_reinit_error=True,
        )
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        bar = remote_tqdm.remote(total=len(self._df_configs))
        futures = [
            self._run.remote(
                index=index,
                config=ConfigDSI(
                    c=row[Param.c],
                    a=row[Param.a],
                    k=int(row[Param.k]),
                    num_target_servers=row[Param.num_target_servers],
                ),
                bar=bar,
            )
            for index, row in self._df_configs.iterrows()
        ]
        self._results_raw = ray.get(futures)
        bar.close.remote()
        ray.shutdown()
        self._merge_results()
        return self.df_results

    @staticmethod
    @ray.remote
    def _run(
        index: int, config: ConfigDSI, bar: tqdm_ray.tqdm
    ) -> tuple[int, dict[str, float]]:
        latencies: dict[str, float] = worker_run_simuls(config)
        bar.update.remote(1)
        return index, latencies

    def _merge_results(self) -> None:
        for i, res in self._results_raw:
            for key, val in res.items():
                self.df_results.at[i, key] = val


def worker_run_simuls(config: ConfigDSI) -> dict[str, float]:
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
