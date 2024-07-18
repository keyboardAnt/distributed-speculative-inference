import itertools
import logging
from contextlib import suppress

import numpy as np
import pandas as pd

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap
from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.types.exception import NumOfTargetServersInsufficientError
from dsi.types.name import Param

log = logging.getLogger(__name__)


def get_df_heatmap_params(config: ConfigHeatmap) -> pd.DataFrame:
    """
    Generate a pandas dataframe with all the configurations of c, a, k that are valid
    for DSI.
    """
    c_vals: np.ndarray = np.linspace(0, 1, config.ndim + 1)
    c_vals = np.where(c_vals <= config.c_min, config.c_min, c_vals)
    a_vals: np.ndarray = np.linspace(0, 1, config.ndim + 1)
    a_vals = np.where(a_vals <= config.a_min, config.a_min, a_vals)
    a_vals = np.where(a_vals >= config.a_max, config.a_max, a_vals)
    ks_space: np.ndarray = np.arange(
        1, 1 + config.k_step * config.ndim, config.k_step, dtype=int
    )
    df_params: pd.DataFrame = pd.DataFrame(
        list(itertools.product(c_vals.tolist(), a_vals.tolist(), ks_space.tolist())),
        columns=[Param.c, Param.a, Param.k],
    )
    df_params[Param.num_target_servers] = config.num_target_servers
    df_params[Param.k] = df_params[Param.k].astype(int)
    df_params[Param.num_target_servers] = df_params[Param.num_target_servers].astype(
        int
    )
    df_params = df_params.drop_duplicates()

    def is_valid_config_dsi(row: pd.Series) -> bool:
        with suppress(NumOfTargetServersInsufficientError):
            ConfigDSI(**row.to_dict())
            return True
        return False

    is_valid_mask = df_params.apply(is_valid_config_dsi, axis=1)
    df_params = df_params[is_valid_mask]
    log.info(f"Number of valid configurations: {len(df_params)}")
    return df_params
