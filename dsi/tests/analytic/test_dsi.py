import pytest

from dsi.analytic.dsi import RunDSI
from dsi.config import ConfigRunDSI
from dsi.types.exceptions import AcceptanceRateError, WaitsOnTargetServerError
from dsi.types.results import Result


@pytest.fixture(params=[0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
def a(request):
    return request.param


@pytest.fixture(params=[1, 2, 1000])
def S(request):
    return request.param


@pytest.fixture(params=[1, 2, 1000])
def num_repeats(request):
    return request.param


@pytest.mark.parametrize(
    "c,failure_cost,k,num_target_servers",
    [
        (0.1, 1.0, 5, 1),
        (0.5, 1.0, 1, 1),
        (0.01, 2.0, 10, 19),
        (0.011, 1.0, 10, 9),
    ],
)
def test_run_config_dsi_num_target_servers_insufficient(
    c: float,
    failure_cost: float,
    k: int,
    num_target_servers: int,
    a: float,
    S: int,
    num_repeats: int,
):
    """
    Verify that initiating ConfigRunDSI with enough target servers will not raise an error, and will raise an error if there are not enough target servers.
    """
    with pytest.raises(WaitsOnTargetServerError):
        ConfigRunDSI(
            c=c,
            failure_cost=failure_cost,
            a=a,
            S=S,
            num_repeats=num_repeats,
            k=k,
            num_target_servers=num_target_servers,
        )
    ConfigRunDSI(
        c=c,
        failure_cost=failure_cost,
        a=a,
        S=S,
        num_repeats=num_repeats,
        k=k,
        num_target_servers=num_target_servers + 1,
    )


def test_dsi_acceptance_rate():
    with pytest.raises(AcceptanceRateError):
        ConfigRunDSI(a=1.01)


def test_dsi_result_shapes():
    num_repeats: int = 17
    config: ConfigRunDSI = ConfigRunDSI(num_repeats=num_repeats)
    dsi = RunDSI(config)
    res: Result = dsi.run()
    assert len(res.cost_per_run) == num_repeats
