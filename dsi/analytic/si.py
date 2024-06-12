import numpy as np

from dsi.schemas.results import ResultSI
from dsi.schemas.run import Run


class AcceptanceRateError(Exception):
    """Raised when the acceptance rate is not in [0, 1]."""

    pass


class RunSI(Run):
    """Run speculative inference."""

    def _get_empty_result(self) -> ResultSI:
        return ResultSI()

    def run(self) -> ResultSI:
        if self.config.a > 1:
            raise AcceptanceRateError(f"Invalid acceptance rate: {self.config.a}")

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
            num_iters: int = 0
            while total_toks < self.config.S:
                num_accepted: int = get_num_accepted()
                total_toks += num_accepted + 1
                total_cost += self.config.k * self.config.c + self.config.failure_cost
                num_iters += 1
            self.result.cost_per_run.append(total_cost)
            self.result.num_iters_per_run.append(num_iters)
        return self.result
