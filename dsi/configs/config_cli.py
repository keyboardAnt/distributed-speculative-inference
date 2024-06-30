import enum

from hydra.core.config_store import ConfigStore
from pydantic import BaseModel, Field

from dsi.configs.config_heatmap import ConfigHeatmap
from dsi.configs.config_run import ConfigRun, ConfigRunDSI


class RunType(str, enum.Enum):
    analytic = "analytic"
    analytic_heatmap = "analytic_heatmap"
    thread_pool = "thread_pool"


class ConfigCLI(BaseModel):
    run_type: RunType = RunType.analytic
    config_run: ConfigRunDSI | ConfigHeatmap = Field(default_factory=ConfigRunDSI)

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.run_type == RunType.analytic:
            assert isinstance(self.config_run, ConfigRun)
        elif self.run_type == RunType.analytic_heatmap:
            assert isinstance(self.config_run, ConfigHeatmap)
        elif self.run_type == RunType.thread_pool:
            raise NotImplementedError


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigCLI().model_dump())
