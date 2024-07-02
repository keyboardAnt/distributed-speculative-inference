from dsi.configs.run import ConfigRunDSI
from dsi.types.result import Result
from dsi.types.run import Run


class RunDSI(Run):
    """
    RunDSI simulates the DSI algorithm over multiple repeats.
    """

    def __init__(self, config: ConfigRunDSI) -> None:
        """
        NOTE: The input config is of type ConfigRunDSI.
        """
        super().__init__(config)

    def _run_single(self) -> Result:
        total_cost: float = 0
        toks_left: int = self.config.S
        num_iters: int = 0
        while toks_left > 0:
            num_iters += 1
            if toks_left == 1:
                total_cost += self.config.failure_cost
                break
            halt_feasible: bool = toks_left <= self.config.k + 1
            curr_k: int = min(self.config.k, toks_left - 1)
            num_accepted: int = next(self._sampler)
            total_cost += curr_k * self.config.c
            toks_left -= num_accepted + 1
            if (num_accepted < curr_k) or halt_feasible:
                total_cost += self.config.failure_cost
        return Result(
            cost_per_run=[total_cost],
            num_iters_per_run=[num_iters],
        )
