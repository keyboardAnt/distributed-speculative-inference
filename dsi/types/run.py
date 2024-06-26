from dsi.config import ConfigRun
from dsi.types.results import Result
from dsi.utils import set_random_seed


class Run:
    def __init__(self, config: ConfigRun) -> None:
        self.config: ConfigRun = config
        self.result: Result = self._get_empty_result()
        set_random_seed()

    def _get_empty_result(self) -> Result:
        return Result()

    def run(self) -> Result:
        raise NotImplementedError
