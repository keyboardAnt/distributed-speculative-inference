from dsi.configs.vis.heatmap import ConfigVisHeatmap
from dsi.types.df_heatmap import DataFrameHeatmap


class VisHeatmap:
    def __init__(self, df: DataFrameHeatmap) -> None:
        self._df: DataFrameHeatmap = df

    def plot(self, config: ConfigVisHeatmap):
        raise NotImplementedError
