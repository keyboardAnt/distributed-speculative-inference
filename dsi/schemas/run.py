from dataclasses import dataclass, field

from dsi.config import ConfigRun
from dsi.utils import set_random_seed


@dataclass
class Result:
    """
    Args:
        cost_per_run: The total latency for each run
    """

    cost_per_run: list[float] = field(default_factory=list)


class Run:
    def __init__(self, config: ConfigRun) -> None:
        self.config: ConfigRun = config
        self.result: Result = self._get_empty_result()
        set_random_seed()

    def _get_empty_result(self) -> Result:
        return Result()

    def run(self) -> Result:
        raise NotImplementedError
