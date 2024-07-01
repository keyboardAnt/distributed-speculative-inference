from argparse import Namespace
import multiprocessing

from dsi.online.run.run_simulation import restart_draft

args = Namespace(**{
    'dataset': 'mbpp',
    'target_name': 'microsoft/Phi-3-medium-128k-instruct',
    'target_first': 180.28411593660712,
    'target_sub': 100.18306512758136,
    'draft_name': 'microsoft/Phi-3-mini-128k-instruct',
    'draft_first': 57.252046037465334,
    'draft_sub': 3.34112215507775,
    'a_rate': 0.94,
    "max_tokens": 50,
    "sim_target_count": 7,
    "sl": 5,
    "draft_tokens_until_fail": 20,
    "output_dir": "example_output_dir",
    "iter_count": 30,
    "run_type": "federated",
    "total_tokens": 100,
    "wait_for_pipe": 0.1
})


def test_entire_threadpool_used():
    
    total_tokens = args.total_tokens
    sim_shared_dict = multiprocessing.Manager().dict()

    sim_shared_dict["total_tokens"] = total_tokens
    sim_shared_dict["prompt_tokens"] = total_tokens

    # While the stop signal is not received, keep restarting the draft model
    th = restart_draft(args, sim_shared_dict["total_tokens"], sim_shared_dict, args.wait_for_pipe)
    th.join()

    for i in range(args.sim_target_count):
        assert str(i) in sim_shared_dict

def test_single_thread_in_livyatan():
    args.run_type = "livyatan"
    
    total_tokens = args.total_tokens
    sim_shared_dict = multiprocessing.Manager().dict()

    sim_shared_dict["total_tokens"] = total_tokens
    sim_shared_dict["prompt_tokens"] = total_tokens

    # While the stop signal is not received, keep restarting the draft model
    th = restart_draft(args, sim_shared_dict["total_tokens"], sim_shared_dict, args.wait_for_pipe)
    th.join()

    assert "MainThread" in sim_shared_dict