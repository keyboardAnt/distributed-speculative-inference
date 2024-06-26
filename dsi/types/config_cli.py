from enum import Enum

from pydantic import BaseModel, Field

from dsi.types.config_run import ConfigRun


class RunType(str, Enum):
    analytic = "analytic"
    thread_pool = "thread_pool"


class ConfigCLI(BaseModel):
    run_type: RunType = RunType.analytic
    config_run: ConfigRun = Field(default_factory=ConfigRun)
