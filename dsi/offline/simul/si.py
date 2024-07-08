from dsi.types.result import Result
from dsi.types.simul import Simul


class SimulSI(Simul):
    """Simulate speculative inference."""

    def _get_empty_result(self) -> Result:
        return Result()

    def _run_single(self) -> Result:
        total_cost: float = 0
        toks_left: int = self.config.S
        num_iters: int = 0
        while toks_left > 0:
            num_iters += 1
            curr_k: int = min(self.config.k, toks_left - 1)
            total_cost += curr_k * self.config.c + self.config.failure_cost
            num_accepted: int = next(self._sampler)
            toks_left -= num_accepted + 1
        return Result(
            cost_per_run=[total_cost],
            num_iters_per_run=[num_iters],
        )
