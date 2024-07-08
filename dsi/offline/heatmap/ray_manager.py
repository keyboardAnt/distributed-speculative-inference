import logging

import pandas as pd
import ray

from dsi.configs.run.heatmap import ConfigHeatmap
from dsi.offline.heatmap.objective import get_all_latencies
from dsi.offline.heatmap.params import get_df_heatmap_params
from dsi.types.name import Param

log = logging.getLogger(__name__)


class RayManager:
    def __init__(self, config: ConfigHeatmap) -> None:
        self._df_configs: pd.DataFrame = get_df_heatmap_params(config)
        self._results_raw: None | list[tuple[int, dict[str, float]]] = None
        self.df_results: pd.DataFrame = self._df_configs.copy(deep=True)

    def run(self) -> pd.DataFrame:
        # NOTE: Ray discovers and utilizes all available resources by default
        ray.init(
            ignore_reinit_error=True,
        )
        futures = [
            self._process_row.remote(index=index, row=row)
            for index, row in self._df_configs.iterrows()
        ]
        self._results_raw = ray.get(futures)
        ray.shutdown()
        self._merge_results()
        return self.df_results

    @staticmethod
    @ray.remote
    def _process_row(index: int, row: dict) -> tuple[int, dict[str, float]]:
        c: float = row[Param.c]
        a: float = row[Param.a]
        k: int = int(row[Param.k])
        num_target_servers: int = int(row[Param.num_target_servers])
        all_latencies: dict[str, float] = get_all_latencies(
            c=c, a=a, k=k, num_target_servers=num_target_servers
        )
        return index, all_latencies

    def _merge_results(self) -> None:
        for i, res in self._results_raw:
            for key, val in res.items():
                self.df_results.at[i, key] = val