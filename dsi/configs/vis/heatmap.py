from typing import Any, Dict, List, Tuple

import pandas as pd
from pydantic import BaseModel


class ConfigVisHeatmap(BaseModel):
    title: str
    xlabel: str
    ylabel: str
    xticklabels: List[str]
    yticklabels: List[str]
    cmap: str
    annot: bool
    fmt: str
    linewidths: float
    linecolor: str
    square: bool
    cbar: bool
    cbar_kws: Dict[str, Any]
    mask: pd.DataFrame
    figsize: Tuple[float, float]
    dpi: int
    savefig: bool
    savefig_path: str
    savefig_format: str
    savefig_dpi: int
    savefig_bbox_inches: str
    savefig_pad_inches: float
