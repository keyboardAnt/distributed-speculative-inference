from typing import final

from dsi.configs.config_run import ConfigRun
from dsi.types.result import Result
from dsi.utils import set_random_seed


class Run:
    def __init__(self, config: ConfigRun) -> None:
        self.config: ConfigRun = config
        self.result: Result = self._get_empty_result()
        set_random_seed()

    def _get_empty_result(self) -> Result:
        return Result()

    @final
    def run(self) -> Result:
        for _ in range(self.config.num_repeats):
            self.result.extend(self._run_single())
        return self.result

    def _run_single(self) -> Result:
        raise NotImplementedError
