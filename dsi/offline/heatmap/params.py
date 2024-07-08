import itertools
import logging

import numpy as np
import pandas as pd

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap
from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.types.exception import NumOfTargetServersInsufficientError
from dsi.types.name import Param

log = logging.getLogger(__name__)


def is_config_valid(c: float, k: int, verbose: bool) -> bool:
    try:
        ConfigDSI(c=c, k=k)
        return True
    except NumOfTargetServersInsufficientError as e:
        if verbose:
            log.info(e)
        return False


def is_row_valid(row: pd.Series, verbose: bool = False) -> bool:
    return is_config_valid(c=row[Param.c], k=row[Param.k], verbose=verbose)


def get_df_heatmap_params(config: ConfigHeatmap) -> pd.DataFrame:
    """
    Generate a pandas dataframe with all the configurations of c, a, k that are valid
    for DSI.
    """
    if not config:
        config = ConfigHeatmap()
    c_vals: np.ndarray = np.linspace(0, 1, config.ndim + 1)
    c_vals = np.where(c_vals <= config.c_min, config.c_min, c_vals)
    a_vals: np.ndarray = np.linspace(0, 1, config.ndim + 1)
    a_vals = np.where(a_vals <= config.a_min, config.a_min, a_vals)
    a_vals = np.where(a_vals >= config.a_max, config.a_max, a_vals)
    ks_space: np.ndarray = np.arange(
        1, 1 + config.k_step * config.ndim, config.k_step, dtype=int
    )

    # pandas dataframe for c, a, k, num_target_servers
    df_params: pd.DataFrame = pd.DataFrame(
        list(itertools.product(c_vals.tolist(), a_vals.tolist(), ks_space.tolist())),
        columns=[Param.c, Param.a, Param.k],
    )
    df_params[Param.num_target_servers] = config.num_target_servers
    df_params[Param.k] = df_params[Param.k].astype(int)
    df_params = df_params.drop_duplicates()
    is_valid_mask = df_params.apply(is_row_valid, axis=1)
    df_params = df_params[is_valid_mask]
    log.info(f"Number of valid configurations: {len(df_params)}")
    return df_params
