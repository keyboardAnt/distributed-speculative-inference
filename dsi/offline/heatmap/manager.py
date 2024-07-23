import logging

import pandas as pd

from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.types.heatmap.manager import _Manager
from dsi.types.name import Param

log = logging.getLogger(__name__)


class Manager(_Manager):
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
