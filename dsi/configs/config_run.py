from math import ceil

from pydantic import BaseModel, Field

from dsi.types.exception import (
    DrafterSlowerThanTargetError,
    NumOfTargetServersInsufficientError,
)


class ConfigRun(BaseModel):
    """ConfigRun includes all the parameters needed for an experiment.
    Each experiment simulates an algorithm multiple times.
    """

    c: float = Field(
        0.1,
        title="The latency of the drafter",
        description="c=0 requires infinitly many target servers",
        gt=0,
    )
    failure_cost: float = Field(1.0, title="The latency of the target", ge=0)
    a: float = Field(0.9, title="The acceptance rate", ge=0, le=1)
    S: int = Field(1000, title="The number of tokens to generate", ge=1)
    num_repeats: int = Field(
        5, title="The number of times that a single run repeats the simulation", ge=1
    )
    k: int = Field(5, title="Lookahead", ge=0)

    def model_post_init(self, _) -> None:
        """
        Verify that the drafter is not slower than the target.
        """
        if self.c > self.failure_cost:
            msg: str = f"{self.c=} > {self.failure_cost=}"
            raise DrafterSlowerThanTargetError(msg)


class ConfigRunDSI(ConfigRun):
    """
    ConfigRunDSI extends ConfigRun for DSI.
    """

    num_target_servers: None | int = Field(
        7,
        title="The number of target servers",
        description="The maximal number of target servers at any point in time. None means infinity.",
        ge=1,
    )

    def model_post_init(self, _) -> None:
        """
        Verify that there are enough target servers so that threads never wait to be executed.
        NOTE: `None` number of target servers means infinity.
        """
        if self.num_target_servers is None:
            return
        num_target_servers_required: int = ceil(
            self.failure_cost / (max(1, self.k) * self.c)
        )
        if self.num_target_servers < num_target_servers_required:
            msg: str = (
                f"num_target_servers={self.num_target_servers} < num_target_servers_required={num_target_servers_required}"
            )
            raise NumOfTargetServersInsufficientError(msg)
