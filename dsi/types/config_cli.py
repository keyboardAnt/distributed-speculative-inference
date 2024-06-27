import enum

from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field

from dsi.types.config_run import ConfigRun


class RunType(enum.Enum):
    analytic = 1
    analytic_heatmap = 2
    thread_pool = 3


class ConfigCLI(BaseModel):
    run_type: RunType = RunType.analytic
    config_run: ConfigRun = Field(default_factory=ConfigRun)


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigCLI().model_dump())
