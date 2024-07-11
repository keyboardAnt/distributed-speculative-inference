from typing import Any, Callable, Literal

import pandas as pd
from pydantic import BaseModel


class ConfigPlotHeatmap(BaseModel):
    col_speedup: str
    mask_fn: Callable[[pd.DataFrame], pd.Series]
    levels_step: None | float = None
    vmax: None | float = None
    pink_idx_side: Literal["left", "right"] = "left"

    def model_post_init(self, __context: Any) -> None:
        return super().model_post_init(__context)
