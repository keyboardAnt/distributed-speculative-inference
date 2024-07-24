import logging
from abc import ABC, abstractmethod
from typing import final

import pandas as pd

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap
from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.types.name import Param

log = logging.getLogger(__name__)


class _Manager(ABC):
    @final
    def __init__(self, config_heatmap: ConfigHeatmap, simul_defaults: ConfigDSI):
        # NOTE: Initializing (e.g. `ConfigHeatmap(**config_heatmap)`) because, in
        # runtime, the type of the given objects is a Hydra's class rather than
        # `ConfigHeatmap` or `ConfigDSI`.
        if not isinstance(config_heatmap, ConfigHeatmap):
            config_heatmap = ConfigHeatmap(**config_heatmap)
        if not isinstance(simul_defaults, ConfigDSI):
            simul_defaults = ConfigDSI(**simul_defaults)
        self._config_heatmap: ConfigHeatmap = config_heatmap
        self._df_config_heatmap: pd.DataFrame = config_heatmap.to_dataframe()
        self._simul_defaults: ConfigDSI = simul_defaults
        self._results_raw: None | list[tuple[int, dict[str, float]]] = None
        self.df_results: pd.DataFrame = self._df_config_heatmap.copy(deep=True)

    @abstractmethod
    def run(self) -> pd.DataFrame:
        raise NotImplementedError

    @final
    @staticmethod
    def _update_config_simul(config_simul: ConfigDSI, row: pd.Series) -> ConfigDSI:
        """
        Update the given `config_simul` with the values from the given `row`.
        """
        config_simul.c = row[Param.c]
        config_simul.a = row[Param.a]
        config_simul.k = int(row[Param.k])
        config_simul.num_target_servers = row[Param.num_target_servers]
        return config_simul

    @final
    def _merge_results(self) -> None:
        for i, res in self._results_raw:
            for key, val in res.items():
                self.df_results.at[i, key] = val
