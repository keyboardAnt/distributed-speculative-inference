from dsi.analytic.common import get_num_accepted_drafts
from dsi.configs.config_run import ConfigRunDSI
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
        print(f"{self.config=}")
        result = Result()
        total_cost: float = 0
        toks_left: int = self.config.S
        num_iters: int = 0
        while toks_left > 0:
            print("------------START---", f"{num_iters=}")
            print(f"{toks_left=}")
            print(f"BEFORE: {total_cost=}")
            num_iters += 1
            if toks_left == 1:
                total_cost += self.config.failure_cost
                break
            curr_k: int = min(self.config.k, toks_left - 1)
            num_accepted: int = get_num_accepted_drafts(
                acceptance_rate=self.config.a, lookahead=curr_k
            )
            print(f"{curr_k=}, {num_accepted=}")
            # nonsi_fwds: int = min(toks_left, num_accepted + 1)
            # cost_nonsi: float = nonsi_fwds * self.config.failure_cost
            cost_dsi = curr_k * self.config.c
            toks_left -= num_accepted
            if num_accepted < curr_k:
                print(f"{num_accepted=}, {curr_k=}, {num_accepted < curr_k=}")
                cost_dsi += self.config.failure_cost
                toks_left -= 1
            if toks_left == 0:
                cost_dsi += self.config.failure
            # print(f"{cost_nonsi=}, {cost_dsi=}")
            # total_cost += min(cost_nonsi, cost_dsi)
            total_cost += cost_dsi
            print(f"AFTER: {total_cost=}")
        result.cost_per_run = [total_cost]
        result.num_iters_per_run = [num_iters]
        return result
