from dsi.analytic.common import get_num_accepted_tokens
from dsi.types.result import ResultSI
from dsi.types.run import Run


class RunSI(Run):
    """Run speculative inference."""

    def _get_empty_result(self) -> ResultSI:
        return ResultSI()

    def _run_single(self) -> ResultSI:
        result = ResultSI()
        total_cost: float = 0
        total_toks: int = 0
        num_iters: int = 0
        while total_toks < self.config.S:
            num_accepted: int = get_num_accepted_tokens(
                acceptance_rate=self.config.a, lookahead=self.config.k
            )
            total_toks += num_accepted + 1
            total_cost += self.config.k * self.config.c + self.config.failure_cost
            num_iters += 1
        result.cost_per_run = [total_cost]
        result.num_iters_per_run = [num_iters]
        return result
