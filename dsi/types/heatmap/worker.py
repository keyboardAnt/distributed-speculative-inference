from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.types.result import ResultWorker


class _Worker:
    def run(self, index: int, config: ConfigDSI) -> ResultWorker:
        """
        NOTE: This function is a workaround to allow using the index of the dataframe.
        """
        return index, self._run(config)

    def _run(self, config: ConfigDSI) -> ResultWorker:
        """
        Executes all the simulations and averages the results over their repeats.
        """
        raise NotImplementedError
