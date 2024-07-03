import logging
import multiprocessing
import time

from tqdm import tqdm

from dsi.configs.config_run import ConfigRunDSISim
from dsi.online.run.core import restart_draft
from dsi.types.result import Result
from dsi.types.run import Run


class RunSim(Run):
    """Run simulated multi-threaded speculative inference."""

    def __init__(self, config: ConfigRunDSISim) -> None:
        super().__init__(config)

    def _get_empty_result(self) -> Result:
        return Result()

    def _run_single(self) -> Result:
        cost_per_run = []
        num_iters_per_run = []
        # Run the simulation {config.num_repeats} times
        for _ in tqdm(range(self.config.num_repeats)):
            total_tokens = self.config.total_tokens
            sim_shared_dict = multiprocessing.Manager().dict()

            sim_shared_dict["total_tokens"] = total_tokens
            sim_shared_dict["prompt_tokens"] = total_tokens

            start_time = time.time()
            iter_till_stop = 0

            # While the stop signal is not received, keep restarting the draft model
            while "stop" not in sim_shared_dict:
                th = restart_draft(
                    self.config,
                    sim_shared_dict["total_tokens"],
                    sim_shared_dict,
                    self.config.wait_for_pipe,
                )
                th.join()
                iter_till_stop += 1
                sim_shared_for_check = {k: v for k, v in sim_shared_dict.items()}
                logging.error(f"{sim_shared_for_check=}")
            inference_time = time.time() - start_time

            # Remove the extra time from the final inference time count
            inference_time = inference_time - iter_till_stop * self.config.wait_for_pipe

            cost_per_run.append(inference_time)
            num_iters_per_run.append(iter_till_stop)

        return Result(
            cost_per_run=cost_per_run,
            num_iters_per_run=num_iters_per_run,
        )
