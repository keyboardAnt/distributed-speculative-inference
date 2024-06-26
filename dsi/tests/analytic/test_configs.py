import pytest
from pydantic import ValidationError

from dsi.types.config_run import ConfigRun, ConfigRunDSI


def test_si_acceptance_rate():
    with pytest.raises(ValidationError):
        ConfigRun(a=1.01)


def test_dsi_acceptance_rate():
    with pytest.raises(ValidationError):
        ConfigRunDSI(a=1.01)
