import logging

import pandas as pd
import ray
from ray.experimental import tqdm_ray

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap
from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.offline.heatmap.ray_worker import RayWorker
from dsi.types.name import Param

log = logging.getLogger(__name__)


class RayManager:
    def __init__(self, config_heatmap: ConfigHeatmap, simul_defaults: ConfigDSI):
        # NOTE: Initializing (e.g. `ConfigHeatmap(**config_heatmap)`) because, in
        # runtime, the type of the given objects is a Hydra's class rather than
        # `ConfigHeatmap` or `ConfigDSI`.
        self._df_config_heatmap: pd.DataFrame = ConfigHeatmap(
            **config_heatmap
        ).to_dataframe()
        self._simul_defaults: ConfigDSI = ConfigDSI(**simul_defaults)
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
            config: ConfigDSI = self._update_config_simul(
                config_simul=self._simul_defaults.model_copy(deep=True), row=row
            )
            futures.append(RayWorker.run.remote(index, config))
        bar.update.remote(1)
        self._results_raw = ray.get(futures)
        bar.close.remote()
        ray.shutdown()
        self._merge_results()
        return self.df_results

    def _update_config_simul(
        self, config_simul: ConfigDSI, row: pd.Series
    ) -> ConfigDSI:
        """
        Update the given `config_simul` with the values from the given `row`.
        """
        config_simul.c = row[Param.c]
        config_simul.a = row[Param.a]
        config_simul.k = int(row[Param.k])
        config_simul.num_target_servers = row[Param.num_target_servers]
        return config_simul

    def _merge_results(self) -> None:
        for i, res in self._results_raw:
            for key, val in res.items():
                self.df_results.at[i, key] = val
