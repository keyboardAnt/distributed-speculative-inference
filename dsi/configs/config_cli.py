import enum

from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field

from dsi.configs.config_heatmap import ConfigHeatmap
from dsi.configs.config_run import ConfigRunDSI


class RunType(str, enum.Enum):
    offline = "offline"
    offline_heatmap = "offline_heatmap"
    online = "online"


class ConfigCLI(BaseModel):
    run_type: RunType = RunType.offline
    config_run: ConfigRunDSI = Field(default_factory=ConfigRunDSI)
    config_heatmap: ConfigHeatmap = Field(default_factory=ConfigHeatmap)


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigCLI().model_dump())
