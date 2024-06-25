import pytest

from dsi.config import ConfigRunDSI, WaitsOnTargetServerError


@pytest.mark.parametrize("a", [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0])
def test_dsi_run_config_num_target_servers(a: float) -> None:
    """
    Verify that initiating ConfigRunDSI without enough target servers throws an error.
    """
    with pytest.raises(WaitsOnTargetServerError):
        ConfigRunDSI(
            c=0.5,
            failure_cost=1.0,
            a=a,
            S=1000,
            num_repeats=5,
            k=5,
            num_target_servers=7,
        )
