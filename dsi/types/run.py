from typing import Generator, final

from dsi.configs.run.algo import ConfigSI
from dsi.offline.run.common import generate_num_accepted_drafts
from dsi.types.result import Result
from dsi.utils import set_random_seed


class Run:
    def __init__(self, config: ConfigSI) -> None:
        self.config: ConfigSI = config
        self.result: Result = self._get_empty_result()
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
        set_random_seed(self.config.random_seed)
        for _ in range(self.config.num_repeats):
            self._init_sampler()
            self.result.extend(self._run_single())
        return self.result

    def _run_single(self) -> Result:
        raise NotImplementedError
