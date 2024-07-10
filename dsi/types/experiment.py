from abc import ABC, abstractmethod
from typing import final

from tqdm import tqdm

from dsi.configs.experiment.base import _ConfigExperiment
from dsi.types.result import ResultSimul, _Result
from dsi.utils import set_random_seed


class _Experiment(ABC):
    def __init__(self, config: _ConfigExperiment) -> None:
        self.config: _ConfigExperiment = config
        self.result: _Result = self._get_empty_result()

    @abstractmethod
    def _get_empty_result(self) -> _Result:
        pass

    def _setup_run(self) -> None:
        """
        A hook for adding additional setup code before running the experiment.
        """
        set_random_seed(self.config.random_seed)

    def _setup_single_repeat(self) -> None:  # noqa: B027
        """
        A hook for adding additional setup code before running a single repeat.
        """
        pass

    @final
    def run(self) -> _Result:
        self._setup_run()
        for _ in tqdm(range(self.config.num_repeats), desc="Repeats"):
            self._setup_single_repeat()
            self.result.extend(self._single_repeat())
        return self.result

    @abstractmethod
    def _single_repeat(self) -> ResultSimul:
        pass
