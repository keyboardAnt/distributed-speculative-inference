from dataclasses import dataclass
from enum import Enum

from hydra.core.config_store import ConfigStore


@dataclass
class ConfigRun:
    c: float = 0.1  # Drafter's latency
    failure_cost: float = 1  # Target's latency
    a: float = 0.9  # Acceptance rate
    S: int = 1000  # Number of tokens to generate
    num_repeats: int = 5  # Number of times to repeat the simulation
    k: int = 5  # Lookahead


class SimulationType(str, Enum):
    analytic = "analytic"
    thread_pool = "thread_pool"


@dataclass
class Config:
    simulation_type: SimulationType = SimulationType.analytic
    config_run: ConfigRun = ConfigRun()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
