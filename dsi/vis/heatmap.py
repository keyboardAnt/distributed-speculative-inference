from dsi.configs.vis.heatmap import ConfigVisHeatmap
from dsi.types.heatmap_df import HeatmapDataFrame


class VisHeatmap:
    def __init__(self, df: HeatmapDataFrame) -> None:
        self._df: HeatmapDataFrame = df

    def plot(self, config: ConfigVisHeatmap):
        raise NotImplementedError
