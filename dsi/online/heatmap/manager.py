import logging

import pandas as pd
from tqdm import tqdm

from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.online.heatmap.worker import WorkerOnline
from dsi.types.heatmap.manager import _Manager

log = logging.getLogger(__name__)


class ManagerOnline(_Manager):
    def run(self) -> pd.DataFrame:
        self._results_raw = []
        # Since the online experiment measures wall time, we avoid using Ray
        # for index, row in self._df_config_heatmap.iterrows():
        for index, row in tqdm(
            self._df_config_heatmap.iterrows(), total=len(self._df_config_heatmap)
        ):
            config: ConfigDSI = self._update_config_simul(
                config_simul=self._simul_defaults.model_copy(deep=True), row=row
            )
            w = WorkerOnline()
            self._results_raw.append(w.run(index=index, config=config))
        self._merge_results()
        return self.df_results
