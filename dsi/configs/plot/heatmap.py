from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel

from dsi.types.exception import InvalidHeatmapColumnError
from dsi.types.name import HeatmapColumn


class ConfigPlotHeatmap(BaseModel):
    col_speedup: str
    mask_fn: Callable[[pd.DataFrame], pd.Series]
    levels_step: None | float = None
    vmax: None | float = None
    pink_idx_side: Literal["left", "right"] = "left"

    def model_post_init(self, __context: Any) -> None:
        if self.col_speedup not in HeatmapColumn.get_all_valid_values():
            raise InvalidHeatmapColumnError(self.col_speedup)
        return super().model_post_init(__context)
