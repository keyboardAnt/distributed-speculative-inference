import itertools
import logging
from contextlib import suppress
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.types.exception import (
    HeatmapConfigInvalidAcceptanceRateRangeError,
    NumOfTargetServersInsufficientError,
)
from dsi.types.name import Param

log = logging.getLogger(__name__)


class ConfigHeatmap(BaseModel):
    online: bool = False
    ndim: int = Field(10, ge=2)
    c_min: float = Field(0.01, title="Minimum drafter latency", ge=0)
    a_min: float = Field(0.01, title="Minimum acceptance rate", ge=0)
    a_max: float = Field(0.99, title="Maximum acceptance rate", le=1)
    k_step: int = Field(1, title="Lookahead step", ge=1)
    num_target_servers: int = Field(7, title="Maximum number of target servers", ge=1)

    def model_post_init(self, __context: Any) -> None:
        if self.a_min > self.a_max:
            raise HeatmapConfigInvalidAcceptanceRateRangeError(
                "Minimum acceptance rate must be less than or equal to maximum"
                f" acceptance rate. Received: min={self.a_min}, max={self.a_max}"
            )
        return super().model_post_init(__context)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Generate a pandas dataframe with all the configurations in the cartesian product
        that has enough target servers.
        """
        c_vals: np.ndarray = np.linspace(0, 1, self.ndim + 1)
        c_vals = np.where(c_vals <= self.c_min, self.c_min, c_vals)
        a_vals: np.ndarray = np.linspace(0, 1, self.ndim + 1)
        a_vals = np.where(a_vals <= self.a_min, self.a_min, a_vals)
        a_vals = np.where(a_vals >= self.a_max, self.a_max, a_vals)
        ks_space: np.ndarray = np.arange(
            1, 1 + self.k_step * self.ndim, self.k_step, dtype=int
        )
        df_params: pd.DataFrame = pd.DataFrame(
            list(
                itertools.product(c_vals.tolist(), a_vals.tolist(), ks_space.tolist())
            ),
            columns=[Param.c, Param.a, Param.k],
        )
        df_params[Param.num_target_servers] = self.num_target_servers
        df_params[Param.k] = df_params[Param.k].astype(int)
        df_params[Param.num_target_servers] = df_params[
            Param.num_target_servers
        ].astype(int)
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
