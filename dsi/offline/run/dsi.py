from dsi.configs.run.run import ConfigRunDSI
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
        cost: float = 0
        toks: int = 0
        iters: int = 0
        while toks < self.config.S:
            new_toks: int = 0
            si: float = 0
            while True:
                iters += 1
                curr_k: int = min(
                    self.config.k, self.config.S - 1 - toks - new_toks
                )  # curr_k >= 0
                accepted: int = min(curr_k, next(self._sampler))  # accepted >= 0
                new_toks += accepted
                si += curr_k * self.config.c
                if (curr_k == 0) or (accepted < curr_k):
                    # At least one draft is rejected
                    si += self.config.failure_cost
                    new_toks += 1
                    break
            nonsi: float = new_toks * self.config.failure_cost
            cost += min(si, nonsi)
            toks += new_toks

        return Result(
            cost_per_run=[cost],
            num_iters_per_run=[iters],
        )
