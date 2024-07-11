import numpy as np
import pandas as pd


def mask_ones(df: pd.DataFrame) -> pd.Series:
    """
    Return a mask of ones with the same shape as the input DataFrame.
    """
    return np.ones_like(df.index, dtype=bool)
