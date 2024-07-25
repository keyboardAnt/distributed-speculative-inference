import logging

import pandas as pd
import ray
from ray.experimental import tqdm_ray

from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.offline.heatmap.worker import Worker
from dsi.types.heatmap.manager import _Manager

log = logging.getLogger(__name__)


class Manager(_Manager):
    def run(self) -> pd.DataFrame:
        self._results_raw = []
        # Offline experiments count time units, so we use Ray to parallelize
        log.info("Initializing Ray for offline heatmap")
        ray.init(ignore_reinit_error=True)
        remote_tqdm = ray.remote(tqdm_ray.tqdm)
        bar: tqdm_ray.tqdm = remote_tqdm.remote(total=len(self._df_config_heatmap))
        futures = []
        for index, row in self._df_config_heatmap.iterrows():
            config: ConfigDSI = self._update_config_simul(
                config_simul=self._simul_defaults.model_copy(deep=True), row=row
            )
            run_remote = ray.remote(Worker.run)
            futures.append(run_remote.remote(Worker(), index, config))
        bar.update.remote(1)
        self._results_raw = ray.get(futures)
        bar.close.remote()
        ray.shutdown()
        self._merge_results()
        return self.df_results
