import ray

from dsi.configs.experiment.simul.offline import ConfigDSI


class _Worker:
    @ray.remote
    def run(self, index: int, config: ConfigDSI) -> tuple[int, dict[str, float]]:
        """
        NOTE: This function is a workaround to allow using the index of the dataframe.
        """
        return index, self._run(config)

    def _run(self, config: ConfigDSI) -> tuple[int, dict[str, float]]:
        """
        Executes all the simulations and averages the results over their repeats.
        """
        raise NotImplementedError
