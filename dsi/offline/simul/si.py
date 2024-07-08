from dsi.types.result import ResultSimul
from dsi.types.simul import Simul


class SimulSI(Simul):
    """Simulate speculative inference."""

    def _get_empty_result(self) -> ResultSimul:
        return ResultSimul()

    def _single_repeat(self) -> ResultSimul:
        total_cost: float = 0
        toks_left: int = self.config.S
        num_iters: int = 0
        while toks_left > 0:
            num_iters += 1
            curr_k: int = min(self.config.k, toks_left - 1)
            total_cost += curr_k * self.config.c + self.config.failure_cost
            num_accepted: int = next(self._sampler)
            toks_left -= num_accepted + 1
        return ResultSimul(
            cost_per_repeat=[total_cost],
            num_iters_per_repeat=[num_iters],
        )
