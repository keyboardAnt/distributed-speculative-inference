from typing import Generator

from dsi.configs.simul.offline import ConfigSI
from dsi.offline.simul.common import generate_num_accepted_drafts
from dsi.types.experiment import _Experiment
from dsi.types.result import ResultSimul


class Simul(_Experiment):
    def __init__(self, config: ConfigSI) -> None:
        self.config: ConfigSI
        super().__init__(config)
        self._sampler: None | Generator = None

    def _get_empty_result(self) -> ResultSimul:
        return ResultSimul()

    def _setup_single_repeat(self) -> None:
        """
        Initiate the sampler generator.
        """
        self._sampler = generate_num_accepted_drafts(
            acceptance_rate=self.config.a,
            lookahead=self.config.k,
            max_num_samples=self.config.S,
        )

    def _single_repeat(self) -> ResultSimul:
        raise NotImplementedError
