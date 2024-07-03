import multiprocessing

from dsi.configs.config_run import ConfigRunDSISim
from dsi.online.run.run import restart_draft

config = ConfigRunDSISim(
    **{
        "failure_cost": 180.28411593660712 / 1000,
        "failure_cost_sub": 100.18306512758136 / 1000,
        "c": 57.252046037465334 / 1000,
        "c_sub": 3.34112215507775 / 1000,
        "a": 0.94,
        "S": 50,
        "num_target_servers": 4,
        "k": 5,
        "maximum_correct_tokens": 20,
        "num_repeats": 30,
        "run_type": "federated",
        "total_tokens": 100,
        "wait_for_pipe": 0.1,
    }
)


def test_entire_threadpool_used():
    total_tokens = config.total_tokens
    sim_shared_dict = multiprocessing.Manager().dict()

    sim_shared_dict["total_tokens"] = total_tokens
    sim_shared_dict["prompt_tokens"] = total_tokens

    # While the stop signal is not received, keep restarting the draft model
    th = restart_draft(
        config, sim_shared_dict["total_tokens"], sim_shared_dict, config.wait_for_pipe
    )
    th.join()

    for i in range(config.num_target_servers):
        assert str(i) in sim_shared_dict


def test_single_thread_in_livyatan():
    config.run_type = "livyatan"

    total_tokens = config.total_tokens
    sim_shared_dict = multiprocessing.Manager().dict()

    sim_shared_dict["total_tokens"] = total_tokens
    sim_shared_dict["prompt_tokens"] = total_tokens

    # While the stop signal is not received, keep restarting the draft model
    th = restart_draft(
        config, sim_shared_dict["total_tokens"], sim_shared_dict, config.wait_for_pipe
    )
    th.join()

    assert "MainThread" in sim_shared_dict
