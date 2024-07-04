import multiprocessing
import time

from dsi.configs.config_run import ConfigRunOnline
from dsi.offline.run.common import generate_num_accepted_drafts
from dsi.online.run.core import restart_draft
from dsi.types.result import Result
from dsi.types.run import Run


class RunOnline(Run):
    """Run simulated multi-threaded speculative inference."""

    def __init__(self, config: ConfigRunOnline) -> None:
        super().__init__(config)

    def _run_single(self) -> Result:
        correct_token_list: list[int] = list(
            generate_num_accepted_drafts(
                acceptance_rate=self.config.a,
                lookahead=self.config.maximum_correct_tokens,
                max_num_samples=self.config.S,
            )
        )
        sim_shared_dict = multiprocessing.Manager().dict()
        sim_shared_dict["total_tokens"] = self.config.total_tokens
        sim_shared_dict["prompt_tokens"] = self.config.total_tokens

        iter_till_stop = 0
        start_time = time.time()
        # While the stop signal is not received, keep restarting the draft model
        while "stop" not in sim_shared_dict:
            # sample number of correct tokens
            sim_shared_dict["correct"] = correct_token_list[iter_till_stop]

            th = restart_draft(
                self.config,
                sim_shared_dict["total_tokens"],
                sim_shared_dict,
                self.config.wait_for_pipe,
            )
            th.join()
            iter_till_stop += 1

        inference_time = time.time() - start_time

        # Remove the extra time from the final inference time count
        inference_time -= iter_till_stop * self.config.wait_for_pipe

        return Result(
            cost_per_run=[inference_time],
            num_iters_per_run=[iter_till_stop],
        )
