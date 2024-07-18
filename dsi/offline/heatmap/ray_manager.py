import logging

import pandas as pd
import ray
from ray.experimental import tqdm_ray

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap
from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.offline.heatmap.worker import RayWorker
from dsi.types.name import Param

log = logging.getLogger(__name__)


class RayManager:
    def __init__(self, config_heatmap: ConfigHeatmap, simul_defaults: ConfigDSI):
        # NOTE: Initializing `ConfigHeatmap(**config_heatmap)` because, in runtime, the
        # given object is a Hydra's object rather than `ConfigHeatmap`.
        self._df_config_heatmap: pd.DataFrame = ConfigHeatmap(
            **config_heatmap
        ).to_dataframe()
        self._simul_defaults: ConfigDSI = simul_defaults
        self._results_raw: None | list[tuple[int, dict[str, float]]] = None
        self.df_results: pd.DataFrame = self._df_config_heatmap.copy(deep=True)

    def run(self) -> pd.DataFrame:
        # NOTE: Ray discovers and utilizes all available resources by default
        ray.init(
            ignore_reinit_error=True,
        )
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        bar: tqdm_ray.tqdm = remote_tqdm.remote(total=len(self._df_config_heatmap))
        futures = []
        for index, row in self._df_config_heatmap.iterrows():
            config = ConfigDSI(**self._simul_defaults)
            config.c = row[Param.c]
            config.a = row[Param.a]
            config.k = int(row[Param.k])
            config.num_target_servers = row[Param.num_target_servers]
            futures.append(RayWorker.run.remote(index, config))
        bar.update.remote(1)
        self._results_raw = ray.get(futures)
        bar.close.remote()
        ray.shutdown()
        self._merge_results()
        return self.df_results

    def _merge_results(self) -> None:
        for i, res in self._results_raw:
            for key, val in res.items():
                self.df_results.at[i, key] = val
