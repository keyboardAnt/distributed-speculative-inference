import enum

from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field, FilePath

from dsi.configs.simul.heatmap import ConfigHeatmap
from dsi.configs.simul.offline import ConfigDSI


class RunType(str, enum.Enum):
    offline = "offline"
    offline_heatmap = "offline_heatmap"
    online = "online"


class ConfigCLI(BaseModel):
    type: RunType = RunType.offline
    simul: ConfigDSI = Field(
        default_factory=ConfigDSI, description="Configuration for the simulation"
    )
    heatmap: ConfigHeatmap = Field(
        default_factory=ConfigHeatmap, description="Configuration for the heatmap"
    )
    load_results: None | FilePath = Field(
        None,
        description="Path to the results file to load",
    )


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigCLI().model_dump())
