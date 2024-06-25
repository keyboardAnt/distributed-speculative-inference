import pytest

from dsi.config import ConfigRunDSI, WaitsOnTargetServerError


@pytest.fixture(params=[0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
def a(request):
    return request.param


@pytest.fixture(params=[1, 2, 1000])
def S(request):
    return request.param


def test_dsi_run_config_num_target_servers(a: float, S: int) -> None:
    """
    Verify that initiating ConfigRunDSI without enough target servers throws an error.
    """
    with pytest.raises(WaitsOnTargetServerError):
        ConfigRunDSI(
            c=0.5,
            failure_cost=1.0,
            a=a,
            S=S,
            num_repeats=5,
            k=1,
            num_target_servers=1,
        )
