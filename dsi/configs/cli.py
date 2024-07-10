import enum

from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field, FilePath

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap
from dsi.configs.experiment.simul.offline import ConfigDSI


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
    load_csv: None | FilePath = Field(
        None,
        description=(
            "Path to the results CSV file to load."
            " If None, a new experiment will be run."
        ),
    )


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigCLI().model_dump())
