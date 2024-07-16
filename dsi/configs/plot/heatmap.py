from typing import Any

from pydantic import Field

from dsi.configs.plot.base import ConfigPlot
from dsi.types.exception import ConfigPlotHeatmapInvalidLevelsStepError
from dsi.types.name import HeatmapColumn


class ConfigPlotHeatmap(ConfigPlot):
    val_col: HeatmapColumn
    levels_step: None | float = Field(1.0, le=1.0)
    vmax: None | float = None

    def model_post_init(self, __context: Any) -> None:
        if ((1 / self.levels_step) % 1) != 0:
            raise ConfigPlotHeatmapInvalidLevelsStepError(self.levels_step)
        return super().model_post_init(__context)
