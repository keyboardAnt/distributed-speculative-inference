from typing import Callable, Literal

import pandas as pd
from pydantic import BaseModel


class ConfigVisHeatmap(BaseModel):
    col_speedup: str
    mask_fn: Callable[[pd.DataFrame], pd.Series]
    levels_step: None | float = None
    vmax: None | float = None
    pink_idx_side: Literal["left", "right"] = "left"
