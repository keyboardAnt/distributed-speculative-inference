import itertools
import logging

import numpy as np
import pandas as pd

from dsi.types.config_run import ConfigRunDSI
from dsi.types.exception import NumOfTargetServersInsufficientError

log = logging.getLogger(__name__)


class Param:
    c = "c"
    a = "a"
    k = "k"


def is_config_valid(c: float, k: int, verbose: bool) -> bool:
    try:
        ConfigRunDSI(c=c, k=k)
        return True
    except NumOfTargetServersInsufficientError as e:
        if verbose:
            log.info(e)
        return False


def is_row_valid(row: pd.Series, verbose: bool = False) -> bool:
    return is_config_valid(c=row[Param.c], k=row[Param.k], verbose=verbose)


def get_df_confs() -> pd.DataFrame:
    """
    Generate a pandas dataframe with all the configurations of c, a, k that are valid for DSI.
    """
    ndim: int = 200
    c_vals: list[float] = np.linspace(0, 1, ndim + 1).tolist()
    c_vals[0] = 0.01
    a_vals: list[float] = np.linspace(0, 1, ndim + 1).tolist()
    a_vals[0] = 0.01
    a_vals[-1] = 0.99
    k_step: int = 1
    ks_space: list[int] = np.arange(1, 1 + k_step * ndim, k_step, dtype=int).tolist()

    # pandas dataframe for c, a, k
    df_params: pd.DataFrame = pd.DataFrame(
        list(itertools.product(c_vals, a_vals, ks_space)),
        columns=[Param.c, Param.a, Param.k],
    )
    df_params[Param.k] = df_params[Param.k].astype(int)
    df_params = df_params.drop_duplicates()
    is_valid_mask = df_params.apply(is_row_valid, axis=1)
    df_params = df_params[is_valid_mask]
    log.info(f"Number of valid configurations: {len(df_params)}")
    return df_params
