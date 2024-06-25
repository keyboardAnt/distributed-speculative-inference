from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore


@dataclass
class ConfigRun:
    """
    ConfigRun includes all the parameters needed for an experiment.
    Each experiment simulates an algorithm multiple times.
    """

    c: float = 0.1  # Drafter's latency
    failure_cost: float = 1  # Target's latency
    a: float = 0.9  # Acceptance rate
    S: int = 1000  # Number of tokens to generate
    num_repeats: int = 5  # Number of times to repeat the simulation
    k: int = 5  # Lookahead


class WaitsOnTargetServerError(Exception):
    def __init__(
        self,
        message="The current analysis supports only simples cases where there are no waits on target servers. For every k drafts that are ready for verification, there must be an idle target server.",
    ):
        super().__init__(message)


@dataclass
class ConfigRunDSI(ConfigRun):
    """
    ConfigRunDSI is the same as ConfigRun but for DSI.
    """

    num_target_servers: int = (
        7  # The maximal number of target servers at any point in time
    )

    def verify_no_waits_on_target_servers(self) -> None:
        """
        Verify that there are enough target servers so that threads never wait to be executed.
        """
        num_target_servers_required: int = self.failure_cost / (self.k * self.c)
        if self.num_target_servers < num_target_servers_required:
            raise WaitsOnTargetServerError()

    def __post_init__(self) -> None:
        self.verify_no_waits_on_target_servers()


class SimulationType(str, Enum):
    analytic = "analytic"
    thread_pool = "thread_pool"


@dataclass
class Config:
    simulation_type: SimulationType = SimulationType.analytic
    config_run: ConfigRun = ConfigRun()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
