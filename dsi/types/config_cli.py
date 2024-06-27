import enum

from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field

from dsi.types.config_run import ConfigRun


class RunType(str, enum.Enum):
    analytic = "analytic"
    analytic_heatmap = "analytic_heatmap"
    thread_pool = "thread_pool"


class ConfigCLI(BaseModel):
    run_type: RunType = RunType.analytic
    config_run: ConfigRun = Field(default_factory=ConfigRun)


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigCLI().model_dump())
