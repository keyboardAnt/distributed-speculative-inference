import contextlib
import functools
import multiprocessing
import os
import queue
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pipe
from threading import current_thread
from typing import Callable

from dsi.configs.experiment.simul.online import ConfigDSIOnline, SimulType


def terminate_process(cur_pipe, sim_executor):
    """
    killing the current threadpool and process,
    """
    if sim_executor is not None:
        sim_executor.shutdown(wait=False, cancel_futures=True)

    cur_pid1 = cur_pipe[0].recv()
    os.kill(cur_pid1, signal.SIGTERM)


def fix_history(total_tokens, correct, sim_shared_dict, cur_pipe, sim_executor):
    """
    Simulate fixing the history
    """

    sim_shared_dict["total_tokens"] = total_tokens + correct + 1

    terminate_process(cur_pipe, sim_executor)


def get_target_time(args, visited):
    """
    Get the target inference time
    """
    if not visited:
        return args.failure_cost
    return args.failure_cost_sub


def get_current_thread_name():
    """
    Get the current thread name in the threadpool. If thread pool does not exist,
    then current process name will be returned instead.
    """
    cur_thread_name = current_thread().getName()
    model_id = (
        cur_thread_name.split("_")[1] if "_" in cur_thread_name else cur_thread_name
    )
    return model_id


def call_target_actual(
    args, draft_tokens, total_tokens, sim_shared_dict, cur_pipe, sim_executor
):
    model_id = get_current_thread_name()

    visited = model_id in sim_shared_dict
    cur_target_time = get_target_time(args, visited)
    if not visited:
        sim_shared_dict[model_id] = True

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

    # First for target input, second for extra target token
    if res_dict["correct"] <= res_dict["draft_tokens"]:
        # I have "correct" correct token, plus 1
        # ONLY {correct} are correct, need to fix the history
        fix_history(
            res_dict["total_tokens"],
            res_dict["correct"],
            res_dict["sim_shared_dict"],
            res_dict["cur_pipe"],
            res_dict["sim_executor"],
        )
    else:
        # ALL CORRECT with {total_tokens + draft_tokens}

        res_dict["total_tokens"] += res_dict["correct"] + 1

        if res_dict["total_tokens"] >= args.max_tokens:
            # MAX TOKENS REACHED
            res_dict["sim_shared_dict"]["stop"] = True
            terminate_process(res_dict["cur_pipe"], res_dict["sim_executor"])


def call_target(
    args: ConfigDSIOnline,
    total_tokens,
    draft_tokens,
    sim_shared_dict,
    sim_executor,
    cur_pipe,
):
    """
    Call the target in a separate thread
    """
    call_target_actual_partial: Callable = functools.partial(
        call_target_actual,
        args=args,
        total_tokens=total_tokens,
        draft_tokens=draft_tokens,
        sim_shared_dict=sim_shared_dict,
        cur_pipe=cur_pipe,
        sim_executor=sim_executor,
    )
    match args.simul_type:
        case SimulType.DSI:
            with contextlib.suppress(RuntimeError):
                # If the executor was shutdown, new submissions will raise a
                # RuntimeError. Otherwise, the executor has not been shutdown and we
                # can submit the task.
                target_future = sim_executor.submit(call_target_actual_partial)
                target_future.add_done_callback(
                    functools.partial(target_done_callback, args)
                )
        case SimulType.SI:
            target_res = call_target_actual_partial()
            target_done_callback(args, target_res)
        case _:
            raise ValueError(f"Invalid run type {args.simul_type}")


def run_generate(
    args: ConfigDSIOnline, total_tokens, sim_shared_dict, cur_pipe, wait_for_pipe
):
    """
    Run the generation process for the draft model.
    The target model is called every k tokens.
    """
    # if has to wait for the pipe to send the pid, wait for wait_for_pipe seconds
    if wait_for_pipe:
        time.sleep(wait_for_pipe)

    if args.simul_type == SimulType.DSI:
        # create a LIFO threadpool
        sim_executor = ThreadPoolExecutor(max_workers=args.num_target_servers)
        sim_executor._work_queue = queue.LifoQueue()
    else:
        sim_executor = None

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
        current_draft_time: float = args.c_sub
        if is_first_token and draft_tokens == 0:
            current_draft_time = args.c
        time.sleep(current_draft_time)
        draft_tokens += 1
        # call the target model every 'lookahead' tokens

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
