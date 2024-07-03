from math import ceil
from typing import Literal

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
    a: float = Field(0.9, title="The acceptance rate", ge=0, le=1)
    k: int = Field(5, title="Lookahead", ge=1)
    failure_cost: float = Field(1.0, title="The latency of the target", ge=0)
    S: int = Field(1000, title="The number of tokens to generate", ge=1)
    num_repeats: int = Field(
        5, title="The number of times that a single run repeats the simulation", ge=1
    )

    def model_post_init(self, __context) -> None:
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
        description=(
            "The maximal number of target servers at any point in time."
            " None means infinity."
        ),
        ge=1,
    )

    def model_post_init(self, __context) -> None:
        """
        Verify that there are enough target servers so that threads never wait
        to be executed.
        NOTE: `None` number of target servers means infinity.
        """
        super().model_post_init(__context)
        if self.num_target_servers is None:
            return
        num_target_servers_required: int = ceil(
            self.failure_cost / (max(1, self.k) * self.c)
        )
        if self.num_target_servers < num_target_servers_required:
            msg: str = (
                f"num_target_servers={self.num_target_servers}"
                " < num_target_servers_required={num_target_servers_required}"
            )
            raise NumOfTargetServersInsufficientError(msg)

class ConfigRunDSISim(ConfigRunDSI):
    c_sub: float = Field(
        0.01,
        title="The latency of the drafter subsequent tokens",
        description="c=0 requires infinitly many target servers",
        gt=0,
    )
    failure_cost_sub: float = Field(0.1, title="The latency of the target subsequent tokens", ge=0)
    total_tokens: int = Field(100, title="The number of tokens in the prompt", ge=0)
    wait_for_pipe: float = Field(0.1, title="Time to wait for pid to be sent via the pipe, allowing the process to be killed.")
    run_type: Literal['federated', 'livyatan'] = Field("federated", title="Running federated simulation or regular speculative decoding.")
    maximum_correct_tokens: int = Field(20, title="The maximum number of correct tokens produced by draft", ge=0)

    @property
    def max_tokens(self) -> str:
        return self.total_tokens + self.S

