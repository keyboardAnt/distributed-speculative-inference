import pytest
from pydantic import ValidationError

from dsi.types.config_run import ConfigRun, ConfigRunDSI
from dsi.types.exceptions import NumOfTargetServersInsufficientError


def test_si_acceptance_rate():
    with pytest.raises(ValidationError):
        ConfigRun(a=-0.1)
    with pytest.raises(ValidationError):
        ConfigRun(a=1.01)
    ConfigRun(a=0)
    ConfigRun(a=1)


def test_dsi_acceptance_rate():
    with pytest.raises(ValidationError):
        ConfigRunDSI(a=-1)
    with pytest.raises(ValidationError):
        ConfigRunDSI(a=1.01)
    ConfigRunDSI(a=0)
    ConfigRunDSI(a=1.0)


def test_drafter_latency():
    valid_c: list[float] = [0.01, 0.5, 0.9, 1.0, 2, 10, 1000]
    invalid_c: list[float] = [0, 0.0, -0.1, -1, -1000000]
    for config_run_cls in [ConfigRun, ConfigRunDSI]:
        for c in invalid_c:
            with pytest.raises((ValidationError, NumOfTargetServersInsufficientError)):
                config_run_cls(c=c)
        for c in valid_c:
            try:
                config_run_cls(c=c)
            except NumOfTargetServersInsufficientError:
                pass


def test_num_of_tokens_to_generate():
    for config_run_cls in [ConfigRun, ConfigRunDSI]:
        with pytest.raises(ValidationError):
            config_run_cls(S=-1)
        with pytest.raises(ValidationError):
            config_run_cls(S=0)
        config_run_cls(S=1)


def test_num_of_repeats():
    for config_run_cls in [ConfigRun, ConfigRunDSI]:
        with pytest.raises(ValidationError):
            config_run_cls(num_repeats=-1)
        with pytest.raises(ValidationError):
            config_run_cls(num_repeats=0)
        config_run_cls(num_repeats=1)


def test_lookahead():
    for config_run_cls in [ConfigRun, ConfigRunDSI]:
        with pytest.raises((ValidationError, NumOfTargetServersInsufficientError)):
            config_run_cls(k=-1)
        try:
            config_run_cls(k=0)
            config_run_cls(k=1)
        except NumOfTargetServersInsufficientError:
            pass
