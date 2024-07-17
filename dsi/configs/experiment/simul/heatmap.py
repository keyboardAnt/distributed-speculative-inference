from typing import Any

from pydantic import BaseModel, Field

from dsi.types.exception import HeatmapConfigInvalidAcceptanceRateRangeError


class ConfigHeatmap(BaseModel):
    ndim: int = Field(10, ge=2)
    c_min: float = Field(0.01, title="Drafter latency", ge=0)
    a_min: float = Field(0.01, title="Minimum acceptance rate", ge=0)
    a_max: float = Field(0.99, title="Maximum acceptance rate", le=1)
    k_step: int = Field(1, title="Lookahead step", ge=1)
    num_target_servers: int = Field(7, title="Maximum number of target servers", ge=1)

    def model_post_init(self, __context: Any) -> None:
        if self.a_min > self.a_max:
            raise HeatmapConfigInvalidAcceptanceRateRangeError(
                "Minimum acceptance rate must be less than or equal to maximum"
                f" acceptance rate. Received: min={self.a_min}, max={self.a_max}"
            )
        return super().model_post_init(__context)
