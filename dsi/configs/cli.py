import enum

from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field, FilePath

from dsi.configs.experiment.simul.online import ConfigDSIOnline
from dsi.configs.heatmap import ConfigHeatmap
from dsi.configs.plot.plots import Plots


class RunType(str, enum.Enum):
    sanity = "sanity"
    heatmap = "heatmap"
    table = "table"


class ConfigCLI(BaseModel):
    type: RunType = RunType.sanity
    simul: ConfigDSIOnline = Field(
        default_factory=ConfigDSIOnline, description="Configuration for the simulation"
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
    plots: Plots = Field(
        default_factory=Plots,
        description="Configuration for the plots to generate",
    )


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigCLI().model_dump())
