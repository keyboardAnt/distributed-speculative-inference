from typing import Callable

import pandas as pd
from pydantic import BaseModel

from dsi.types.name import HeatmapColumn
from dsi.types.plot import PinkIndexSide


class ConfigPlotHeatmap(BaseModel):
    col_speedup: HeatmapColumn
    mask_fn: Callable[[pd.DataFrame], pd.Series]
    levels_step: None | float = None
    vmax: None | float = None
    pink_idx_side: PinkIndexSide = PinkIndexSide.left
