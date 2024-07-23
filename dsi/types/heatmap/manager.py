import logging
from abc import ABC, abstractmethod
from typing import final

import pandas as pd
import ray
from ray.experimental import tqdm_ray

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap, ExperimentType
from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.offline.heatmap.worker import Worker
from dsi.online.heatmap.worker import WorkerOnline
from dsi.types.heatmap.worker import _Worker

log = logging.getLogger(__name__)


class _Manager(ABC):
    @final
    def __init__(self, config_heatmap: ConfigHeatmap, simul_defaults: ConfigDSI):
        self._config_heatmap: ConfigHeatmap = config_heatmap
        # NOTE: Initializing (e.g. `ConfigHeatmap(**config_heatmap)`) because, in
        # runtime, the type of the given objects is a Hydra's class rather than
        # `ConfigHeatmap` or `ConfigDSI`.
        if not isinstance(config_heatmap, ConfigHeatmap):
            config_heatmap = ConfigHeatmap(**config_heatmap)
        if not isinstance(simul_defaults, ConfigDSI):
            simul_defaults = ConfigDSI(**simul_defaults)
        self._df_config_heatmap: pd.DataFrame = config_heatmap.to_dataframe()
        self._simul_defaults: ConfigDSI = simul_defaults
        self._results_raw: None | list[tuple[int, dict[str, float]]] = None
        self.df_results: pd.DataFrame = self._df_config_heatmap.copy(deep=True)

    @final
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
            w: _Worker = self._get_worker()
            futures.append(w.run.remote(w, index, config))
        bar.update.remote(1)
        self._results_raw = ray.get(futures)
        bar.close.remote()
        ray.shutdown()
        self._merge_results()
        return self.df_results

    def _get_worker(self) -> _Worker:
        match self._config_heatmap.experiment_type:
            case ExperimentType.OFFLINE:
                return Worker()
            case ExperimentType.ONLINE:
                return WorkerOnline()
            case _:
                raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _update_config_simul(config_simul: ConfigDSI, row: pd.Series) -> ConfigDSI:
        """
        Update the given `config_simul` with the values from the given `row`.
        """
        raise NotImplementedError

    @final
    def _merge_results(self) -> None:
        for i, res in self._results_raw:
            for key, val in res.items():
                self.df_results.at[i, key] = val
