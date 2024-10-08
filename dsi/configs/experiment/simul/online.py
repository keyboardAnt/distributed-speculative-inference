from enum import Enum

from pydantic import Field

from dsi.configs.experiment.simul.offline import ConfigDSI


class SimulType(Enum):
    SI = 1
    DSI = 2


class ConfigDSIOnline(ConfigDSI):
    c_sub: float = Field(
        0.01,
        title="The latency of the drafter subsequent tokens",
        description="c=0 requires infinitly many target servers",
        gt=0,
    )
    failure_cost_sub: float = Field(
        0.1, title="The latency of the target subsequent tokens", ge=0
    )
    total_tokens: int = Field(100, title="The number of tokens in the prompt", ge=0)
    wait_for_pipe: float = Field(
        0.1, title="Wait for pid to be sent via the pipe", ge=0
    )
    simul_type: SimulType = Field(
        SimulType.DSI,
        title="Simulating either DSI or SI.",
    )
    maximum_correct_tokens: int = Field(
        20, title="The maximum number of correct tokens produced by draft", ge=0
    )

    @property
    def max_tokens(self) -> int:
        return self.total_tokens + self.S

    @classmethod
    def from_offline(cls, config: ConfigDSI) -> "ConfigDSIOnline":
        """
        Constructs an instance of ConfigDSIOnline from an instance of ConfigDSI.

        Args:
        config (ConfigDSI): An instance of ConfigDSI from which to initialize the online
        configuration.

        Returns:
        ConfigDSIOnline: An initialized instance of ConfigDSIOnline.
        """
        return cls(
            c=config.c,
            a=config.a,
            k=config.k,
            failure_cost=config.failure_cost,
            S=config.S,
            num_target_servers=config.num_target_servers,
            # Online specific configurations
            c_sub=config.c / 10,
            failure_cost_sub=config.failure_cost / 10,
            total_tokens=100,
            wait_for_pipe=0.1,
            simul_type=SimulType.DSI,
            maximum_correct_tokens=20,
        )
