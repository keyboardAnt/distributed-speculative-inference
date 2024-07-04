import functools
import logging
import multiprocessing
import os
import queue
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pipe
from threading import current_thread

from dsi.configs.config_run import RunType
import numpy as np


def terminate_process(cur_pipe, sim_executor):
    """
    killing the current threadpool and process,
    """
    if sim_executor is not None:
        sim_executor.shutdown(wait=False, cancel_futures=True)

    cur_pid1 = cur_pipe[0].recv()
    _ = cur_pipe[1].recv()

    os.kill(cur_pid1, signal.SIGTERM)


def fix_history(total_tokens, correct, sim_shared_dict, cur_pipe, sim_executor):
    """
    Simulate fixing the history
    """
    total_tokens += correct

    sim_shared_dict["total_tokens"] = total_tokens

    terminate_process(cur_pipe, sim_executor)


def get_target_time(args, visited):
    """
    Get the target inference time
    """
    if not visited:
        return args.failure_cost
    return args.failure_cost_sub


def call_target_actual(
    args, draft_tokens, total_tokens, sim_shared_dict, cur_pipe, sim_executor
):
    cur_thread_name = current_thread().getName()
    model_id = (
        cur_thread_name.split("_")[1] if "_" in cur_thread_name else cur_thread_name
    )

    visited = model_id in sim_shared_dict
    cur_target_time = get_target_time(args, visited)
    if not visited:
        sim_shared_dict[model_id] = True
    logging.error(f"{cur_thread_name} {model_id=} {visited=}")
    time.sleep(cur_target_time)

    return dict(
        correct=sim_shared_dict["correct"],
        draft_tokens=draft_tokens,
        total_tokens=total_tokens,
        sim_shared_dict=sim_shared_dict,
        cur_pipe=cur_pipe,
        sim_executor=sim_executor,
    )


def target_done_callback(args, res):
    if isinstance(res, dict):
        res_dict = res
    else:
        if res.cancelled():
            return
        res_dict = res.result()

    correct = res_dict["correct"]
    draft_tokens = res_dict["draft_tokens"]
    total_tokens = res_dict["total_tokens"]
    sim_shared_dict = res_dict["sim_shared_dict"]
    cur_pipe = res_dict["cur_pipe"]
    sim_executor = res_dict["sim_executor"]

    if correct < draft_tokens:
        logging.error(f"{correct} ARE CORRECT out of {draft_tokens}")
        # I have "correct" correct token, plus 1
        # ONLY {correct} are correct, need to fix the history
        fix_history(total_tokens, correct, sim_shared_dict, cur_pipe, sim_executor)
    else:
        # ALL CORRECT with {total_tokens + draft_tokens}

        total_tokens += correct

        if total_tokens > args.max_tokens:
            # MAX TOKENS REACHED
            logging.error(f"MAX REACHED at {total_tokens}")
            sim_shared_dict["stop"] = True
            terminate_process(cur_pipe, sim_executor)


def call_target(
    args, total_tokens, draft_tokens, sim_shared_dict, sim_executor, cur_pipe
):
    """
    Call the target in a separate thread
    """
    if args.run_type == RunType.DSI:
        logging.error(f"SUBMIT in {draft_tokens}")
        if not sim_executor._shutdown:
            target_future = sim_executor.submit(
                call_target_actual,
                args=args,
                total_tokens=total_tokens,
                draft_tokens=draft_tokens,
                sim_shared_dict=sim_shared_dict,
                cur_pipe=cur_pipe,
                sim_executor=sim_executor,
            )

            target_future.add_done_callback(
                functools.partial(target_done_callback, args)
            )
    else:
        target_res = call_target_actual(
            args=args,
            total_tokens=total_tokens,
            draft_tokens=draft_tokens,
            sim_shared_dict=sim_shared_dict,
            cur_pipe=cur_pipe,
            sim_executor=sim_executor,
        )
        target_done_callback(args, target_res)


def get_draft_time_for_first_token(args, total_tokens):
    return args.c


def get_draft_time_for_sub_token(args, total_tokens, draft_tokens):
    return args.c_sub


# def get_number_of_correct_tokens(a, maximum_correct_tokens):
#     """
#     Sample a random number of correct tokens from the geo(1-a_rate) distribution
#     """
#     np.random.seed(seed=int(time.time()))
#     res_list = np.random.geometric(1 - a, size=1) - 1
#     res = min(res_list[-1], maximum_correct_tokens)

#     return res


def run_generate(args, total_tokens, sim_shared_dict, cur_pipe, wait_for_pipe):
    """
    Run the generation process
    """
    # if has to wait for the pipe to send the pid, wait for wait_for_pipe seconds
    if wait_for_pipe:
        time.sleep(wait_for_pipe)

    if args.run_type == RunType.DSI:
        # create a LIFO threadpool
        logging.error(f"{args.num_target_servers=}")
        sim_executor = ThreadPoolExecutor(max_workers=args.num_target_servers)
        sim_executor._work_queue = queue.LifoQueue()
    else:
        sim_executor = None

    logging.error(f'{sim_shared_dict["correct"]=}')

    draft_tokens = 0

    call_target(
        args=args,
        total_tokens=total_tokens,
        draft_tokens=draft_tokens,
        sim_shared_dict=sim_shared_dict,
        sim_executor=sim_executor,
        cur_pipe=cur_pipe,
    )

    is_first_token = sim_shared_dict["total_tokens"] == sim_shared_dict["prompt_tokens"]
    while True:
        # generating token index {total_tokens+draft_tokens}")

        # if has no history, first token time, else use sub-token time
        if is_first_token and draft_tokens == 0:
            current_draft_time = get_draft_time_for_first_token(args, total_tokens)
        else:
            current_draft_time = get_draft_time_for_sub_token(
                args, total_tokens, draft_tokens
            )

        time.sleep(current_draft_time)

        draft_tokens += 1

        # call the target model every sl tokens.
        if draft_tokens % args.k == 0:
            call_target(
                args=args,
                total_tokens=total_tokens,
                draft_tokens=draft_tokens,
                sim_shared_dict=sim_shared_dict,
                sim_executor=sim_executor,
                cur_pipe=cur_pipe,
            )


def restart_draft(args, total_tokens, sim_shared_dict, wait_for_pipe):
    """
    Start/Restart the main generation process.
    """
    cur_pipe = Pipe()

    th = multiprocessing.Process(
        target=run_generate,
        kwargs=dict(
            args=args,
            total_tokens=total_tokens,
            sim_shared_dict=sim_shared_dict,
            cur_pipe=cur_pipe,
            wait_for_pipe=wait_for_pipe,
        ),
    )

    th.start()

    # send the current pid to the pipe, in order to allow killing the thread from within
    cur_pipe[0].send(th.pid)
    cur_pipe[1].send(th.pid)

    return th
