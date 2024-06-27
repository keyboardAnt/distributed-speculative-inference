from dsi.analytic.common import get_num_accepted_tokens
from dsi.types.config_run import ConfigRunDSI
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

    def run(self) -> Result:
        for _ in range(self.config.num_repeats):
            total_cost: float = 0
            total_toks: int = 0
            while total_toks < self.config.S:
                num_accepted: int = get_num_accepted_tokens(
                    acceptance_rate=self.config.a, lookahead=self.config.k
                )
                while (total_toks < self.config.S) and (num_accepted == self.config.k):
                    nonspec_fwds: int = min(
                        self.config.k + 1, self.config.S - total_toks
                    )
                    total_toks += self.config.k + 1
                    cost_nonspec: float = nonspec_fwds * self.config.failure_cost
                    cost_fed = self.config.k * self.config.c
                    total_cost += min(cost_nonspec, cost_fed)
                    num_accepted = get_num_accepted_tokens(
                        acceptance_rate=self.config.a, lookahead=self.config.k
                    )
                if total_toks < self.config.S:
                    nonspec_fwds: int = min(
                        num_accepted + 1, self.config.S - total_toks
                    )
                    total_toks += num_accepted + 1
                    cost_nonspec: float = nonspec_fwds * self.config.failure_cost
                    cost_fed = self.config.k * self.config.c + self.config.failure_cost
                    total_cost += min(cost_nonspec, cost_fed)
            self.result.cost_per_run.append(total_cost)
        return self.result
