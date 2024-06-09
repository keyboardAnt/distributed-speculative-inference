import numpy as np
from dsi.schemas.results import ResultSI
from dsi.schemas.run import Run


class RunSI(Run):
    """Run speculative inference."""

    def _get_empty_result(self) -> ResultSI:
        return ResultSI()

    def run(self) -> ResultSI:
        assert self.config.a <= 1

        def get_num_accepted() -> int:
            is_accepted: list[bool] = [
                np.random.rand() < self.config.a for _ in range(self.config.k)
            ]
            try:
                return is_accepted.index(False)
            except ValueError:
                return self.config.k

        for _ in range(self.config.num_repeats):
            total_cost: float = 0
            total_toks: int = 0
            while total_toks < self.config.S:
                num_accepted: int = get_num_accepted()
                total_toks += num_accepted + 1
                total_cost += self.config.k * self.config.c + self.config.failure_cost
            self.result.cost_per_run.append(total_cost)
        return self.result
