import uuid
from tqdm import tqdm
import argparse
import logging
import time
import multiprocessing
import json
from simulation_core import restart_draft

def parse_args():
    parser = argparse.ArgumentParser(description="Set variables for the script")
    parser.add_argument("--target_first", type=float, default=5, help="Time fot the target first token.")
    parser.add_argument("--target_sub", type=float, default=1, help="Time fot the target subsequent token.")
    parser.add_argument("--draft_first", type=float, default=1, help="Time fot the draft first token.")
    parser.add_argument("--draft_sub", type=float, default=0.1, help="Time fot the target subsequent token.")
    parser.add_argument("--max_tokens", type=int, default=50, help="Number of tokens to generate")
    parser.add_argument("--a_rate", type=float, default=0.9, help="The model pair acceptance rate.")
    parser.add_argument("--sim_target_count", type=int, default=7, help="Number of targets running in parallel.")
    parser.add_argument("--sl", type=int, default=5, help="Number of tokens to generated with the draft before calling the target model.")
    parser.add_argument("--total_tokens", type=int, default=100, help="Number of initial tokens.")
    parser.add_argument(
        "--draft_tokens_until_fail", type=int, default=20, help="Maximum number of accepted tokens by the target."
    )
    parser.add_argument(
        "--iter_count", type=int, default=30, help="Number of times to run the simulation. "
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Output directory to save the results."
    )
    parser.add_argument(
        "--dataset", type=str, default=None, help="Name of the dataset."
    )
    parser.add_argument(
        "--target_name", type=str, default=None, help="Name of the target model."
    )
    parser.add_argument(
        "--draft_name", type=str, default=None, help="Name of the draft model."
    )
    parser.add_argument(
        "--pair_data", type=str, default=None, help="Model-draft pair data to override the default values."
    )
    parser.add_argument(
        "--run_type", type=str, default="federated", choices=["federated", "livyatan"], help="Running federated simulation or regular speculative decoding."
    )
    parser.add_argument(
        "--wait_for_pipe", type=float, default=0.1, help="Time to wait for pid to be sent via the pipe, allowing the process to be killed."
    )
    args = parser.parse_args()
    return args


def init_params(args):
    if args.pair_data is not None:
        pair_data = json.loads(args.pair_data)
        del args.pair_data

        for k,v in pair_data.items():
            setattr(args, k, v)
    
    # convert time settings to seconds
    for k in [
        "draft_sub",
        "draft_first",
        "target_sub",
        "target_first"
    ]:
        arg_value = getattr(args, k)

        # convert to seconds
        if arg_value > 0:
            setattr(args, k, arg_value / 1000)

    args.max_tokens += args.total_tokens
    run_id = str(uuid.uuid4())

    wait_for_pipe = args.wait_for_pipe
    logging.info(f"BEGIN RUN: {run_id=}, {args=}, \n {wait_for_pipe=}")
    return args, run_id

def main():
    args, run_id = init_params(parse_args())
    
    inference_time_list = []

    # Run the simulation {args.iter_count} times
    for _ in tqdm(range(args.iter_count)):
        total_tokens = args.total_tokens
        sim_shared_dict = multiprocessing.Manager().dict()

        sim_shared_dict["total_tokens"] = total_tokens
        sim_shared_dict["prompt_tokens"] = total_tokens

        start_time = time.time()
        iter_till_stop = 0

        # While the stop signal is not received, keep restarting the draft model
        while "stop" not in sim_shared_dict:
            th = restart_draft(args, sim_shared_dict["total_tokens"], sim_shared_dict, args.wait_for_pipe)
            th.join()
            iter_till_stop += 1
            # sim_shared_for_check = {k:v for k,v in sim_shared_dict.items()}
            # logging.error(f"{sim_shared_for_check=}")
        inference_time = time.time() - start_time
        
        # if waiting for the process to start, remove the extra time from the final inference time count
        inference_time = inference_time - iter_till_stop * args.wait_for_pipe
        
        inference_time_list.append(inference_time)

    result_dict = args.__dict__
    result_dict.update({
        "inference_time": inference_time_list,
        "run_id": run_id
    })

    with open(f"{args.output_dir}/res_{run_id}.json", "w") as f:
        json.dump(result_dict, f)

    logging.info(f"DONE in {inference_time_list}")

if __name__ == "__main__":
    main()
