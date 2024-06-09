import pytest

from dsi.analytic.si import AcceptanceRateError, RunSI
from dsi.config import ConfigRun


def test_si_acceptance_rate() -> None:
    config: ConfigRun = ConfigRun(a=1.01)
    si: RunSI = RunSI(config)
    with pytest.raises(AcceptanceRateError):
        si.run()
