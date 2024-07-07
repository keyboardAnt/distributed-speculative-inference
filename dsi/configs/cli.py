import enum

from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field, FilePath

from dsi.configs.run.heatmap import ConfigHeatmap
from dsi.configs.run.run import ConfigRunDSI


class RunType(str, enum.Enum):
    offline = "offline"
    offline_heatmap = "offline_heatmap"
    online = "online"


class ConfigCLI(BaseModel):
    type: RunType = RunType.offline
    run: ConfigRunDSI = Field(
        default_factory=ConfigRunDSI, description="Configuration for the simulation"
    )
    heatmap: ConfigHeatmap = Field(
        default_factory=ConfigHeatmap, description="Configuration for the heatmap"
    )
    load_csv: None | FilePath = Field(
        None,
        description=(
            "Path to the results CSV file to load."
            " If None, a new experiment will be run."
        ),
    )


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigCLI().model_dump())
