import multiprocessing

from dsi.configs.config_run import ConfigRunOnline, RunType
from dsi.online.run.run import restart_draft
import pytest

@pytest.fixture
def get_config():
    return ConfigRunOnline(
        c=0.05725204603746534, 
        a=0.94, 
        k=5, 
        failure_cost=0.18028411593660712, 
        S=50, 
        num_repeats=30, 
        num_target_servers=4, 
        c_sub=0.0033411221550777503, 
        failure_cost_sub=0.10018306512758136, 
        total_tokens=100, 
        wait_for_pipe=0.1, 
        run_type=RunType.DSI, 
        maximum_correct_tokens=20
    )



def test_entire_threadpool_used(get_config):
    config = get_config()
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


def test_single_thread_in_si(get_config):
    config = get_config()
    config.run_type = RunType.SI

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
