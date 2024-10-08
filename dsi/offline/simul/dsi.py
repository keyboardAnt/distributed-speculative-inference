from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.types.result import ResultSimul
from dsi.types.simul import Simul


class SimulDSI(Simul):
    """
    Simulates the DSI algorithm over multiple repeats.
    """

    def __init__(self, config: ConfigDSI) -> None:
        """
        NOTE: The input config is of type ConfigDSI.
        """
        super().__init__(config)

    def _single_repeat(self) -> ResultSimul:
        cost: float = 0
        toks: int = 0
        iters: int = 0
        verified: bool = False
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
                if (accepted > 0) or (not verified):
                    # Since the logits vector for the first position in the current
                    # iteration is known once the target verifying the previous
                    # iteration is done, we terminate the bad thread immediately.
                    si += curr_k * self.config.c
                if (curr_k == 0) or (accepted < curr_k):
                    # At least one draft is rejected
                    si += self.config.failure_cost
                    new_toks += 1
                    verified = True
                    break
                verified = False
            nonsi: float = new_toks * self.config.failure_cost
            cost += min(si, nonsi)
            toks += new_toks

        return ResultSimul(
            cost_per_repeat=[cost],
            num_iters_per_repeat=[iters],
        )
