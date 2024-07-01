import pandas as pd

from dsi.configs.config_heatmap import ConfigHeatmap
from dsi.offline.heatmap.params import Param, get_df_heatmap_params


def test_get_df_heatmap_params():
    df_params = get_df_heatmap_params()
    assert isinstance(df_params, pd.DataFrame)
    assert df_params[Param.c].dtype == float
    assert df_params[Param.a].dtype == float
    assert df_params[Param.k].dtype == int
    config = ConfigHeatmap()
    assert len(df_params) > 0
    assert all(df_params[Param.c] >= config.c_min)
    assert all(df_params[Param.c] <= 1)
    assert all(df_params[Param.a] >= config.a_min)
    assert all(df_params[Param.a] <= config.a_max)
    assert all(df_params[Param.k] > 0)
    assert all(df_params[Param.k] <= 1 + config.k_step * config.ndim)
    assert all(df_params[Param.k] % config.k_step == 0)
    assert len(df_params) == len(
        df_params.drop_duplicates()
    ), "There are duplicate rows"
