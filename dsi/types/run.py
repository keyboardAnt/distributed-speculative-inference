from typing import Generator, final

from dsi.analytic.common import generate_num_accepted_drafts
from dsi.configs.config_run import ConfigRun
from dsi.types.result import Result
from dsi.utils import set_random_seed


class Run:
    def __init__(self, config: ConfigRun) -> None:
        self.config: ConfigRun = config
        self.result: Result = self._get_empty_result()
        set_random_seed()
        self._sampler: None | Generator = None

    def _get_empty_result(self) -> Result:
        return Result()

    def _init_sampler(self) -> None:
        self._sampler = generate_num_accepted_drafts(
            acceptance_rate=self.config.a,
            lookahead=self.config.k,
            max_num_samples=self.config.S,
        )

    @final
    def run(self) -> Result:
        for i in range(self.config.num_repeats):
            self._init_sampler()
            print(
                f"======================{self.__class__.__name__}======================"
            )
            print(f"{self.config=}")
            print(f"{i=}")
            self.result.extend(self._run_single())
        return self.result

    def _run_single(self) -> Result:
        raise NotImplementedError
