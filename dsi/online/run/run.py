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

    def get_correct_token_list(self):
        """
        Generate random numbers of correct tokens, until the
         total number of tokens is less than S.
        """
        correct_token_list = []
        while sum(correct_token_list) <= self.config.S:
            correct_token_list.append(
                list(
                    generate_num_accepted_drafts(
                        self.config.a, self.config.maximum_correct_tokens, 1
                    )
                )[0]
            )
        return correct_token_list

    def _run_single(self) -> Result:
        correct_token_list = self.get_correct_token_list()

        total_tokens = self.config.total_tokens
        sim_shared_dict = multiprocessing.Manager().dict()

        sim_shared_dict["total_tokens"] = total_tokens
        sim_shared_dict["prompt_tokens"] = total_tokens

        start_time = time.time()
        iter_till_stop = 0

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
        inference_time = inference_time - iter_till_stop * self.config.wait_for_pipe

        return Result(
            cost_per_run=[inference_time],
            num_iters_per_run=[iter_till_stop],
        )
