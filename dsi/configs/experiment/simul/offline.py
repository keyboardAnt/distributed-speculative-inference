from math import ceil

from pydantic import Field

from dsi.configs.experiment.base import _ConfigExperiment
from dsi.types.exception import (
    DrafterSlowerThanTargetError,
    NumOfTargetServersInsufficientError,
)


class ConfigSI(_ConfigExperiment):
    """ConfigSI includes all the parameters needed for an experiment.
    Each experiment simulates an algorithm multiple times.
    """

    c: float = Field(
        0.1,
        title="The latency of the drafter",
        description="c=0 requires infinitly many target servers",
        gt=0,
    )
    a: float = Field(0.9, title="The acceptance rate", ge=0, le=1)
    k: int = Field(5, title="Lookahead", ge=1)
    failure_cost: float = Field(1.0, title="The latency of the target", ge=0)
    S: int = Field(1000, title="The number of tokens to generate", ge=1)

    def model_post_init(self, __context) -> None:
        """
        Verify that the drafter is not slower than the target.
        """
        if self.c > self.failure_cost:
            msg: str = f"{self.c=} > {self.failure_cost=}"
            raise DrafterSlowerThanTargetError(msg)


class ConfigDSI(ConfigSI):
    """
    ConfigDSI extends ConfigSI for DSI.
    """

    num_target_servers: None | int = Field(
        7,
        title="The number of target servers",
        description=(
            "The maximal number of target servers at any point in time."
            " None means infinity."
        ),
        ge=1,
    )

    @property
    def is_sufficient_num_target_servers(self) -> bool:
        """
        Check if the number of target servers is sufficient.
        """
        if self.num_target_servers is None:
            return True
        num_target_servers_required: int = ceil(
            self.failure_cost / (max(1, self.k) * self.c)
        )
        return self.num_target_servers >= num_target_servers_required

    def model_post_init(self, __context) -> None:
        """
        Verify that there are enough target servers so that threads never wait
        to be executed.
        NOTE: `None` number of target servers means infinity.
        """
        super().model_post_init(__context)
        if not self.is_sufficient_num_target_servers:
            msg: str = (
                f"num_target_servers={self.num_target_servers}"
                " < num_target_servers_required={num_target_servers_required}"
            )
            raise NumOfTargetServersInsufficientError(msg)
