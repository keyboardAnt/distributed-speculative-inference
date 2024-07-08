import pytest
from pydantic import ValidationError

from dsi.configs.run.algo import ConfigDSI, ConfigSI
from dsi.types.exception import (
    DrafterSlowerThanTargetError,
    NumOfTargetServersInsufficientError,
)


def test_si_acceptance_rate():
    with pytest.raises(ValidationError):
        ConfigSI(a=-0.1)
    with pytest.raises(ValidationError):
        ConfigSI(a=1.01)
    ConfigSI(a=0)
    ConfigSI(a=1)


def test_dsi_acceptance_rate():
    with pytest.raises(ValidationError):
        ConfigDSI(a=-1)
    with pytest.raises(ValidationError):
        ConfigDSI(a=1.01)
    ConfigDSI(a=0)
    ConfigDSI(a=1.0)


def test_drafter_latency():
    valid_c: list[float] = [0.01, 0.5, 0.9, 1.0, 2, 10, 1000]
    invalid_c: list[float] = [0, 0.0, -0.1, -1, -1000000]
    for config_run_cls in [ConfigSI, ConfigDSI]:
        for c in invalid_c:
            with pytest.raises((ValidationError, NumOfTargetServersInsufficientError)):
                config_run_cls(c=c, failure_cost=c + 0.01)
        for c in valid_c:
            config_run_cls(c=c, failure_cost=c + 0.01, num_target_servers=None)


@pytest.mark.parametrize("c", [0.01, 0.1, 0.8, 0.99, 1.0, 1.01, 2.0, 1000])
@pytest.mark.parametrize("failure_cost", [0.01, 0.1, 0.8, 0.99, 1.0, 1.01, 2.0, 1000])
def test_drafter_latency_vs_target_latency(c: float, failure_cost: float):
    for config_run_cls in [ConfigSI, ConfigDSI]:
        if c <= failure_cost:
            config_run_cls(c=c, failure_cost=failure_cost, num_target_servers=None)
        else:
            with pytest.raises(DrafterSlowerThanTargetError):
                config_run_cls(c=c, failure_cost=failure_cost, num_target_servers=None)


def test_num_of_tokens_to_generate():
    for config_run_cls in [ConfigSI, ConfigDSI]:
        with pytest.raises(ValidationError):
            config_run_cls(S=-1)
        with pytest.raises(ValidationError):
            config_run_cls(S=0)
        config_run_cls(S=1)


def test_num_of_repeats():
    for config_run_cls in [ConfigSI, ConfigDSI]:
        with pytest.raises(ValidationError):
            config_run_cls(num_repeats=-1)
        with pytest.raises(ValidationError):
            config_run_cls(num_repeats=0)
        config_run_cls(num_repeats=1)


def test_lookahead():
    for config_run_cls in [ConfigSI, ConfigDSI]:
        k_invalid: list[int] = [-1, 0]
        for k in k_invalid:
            with pytest.raises(ValidationError):
                config_run_cls(k=k)
        config_run_cls(k=1, num_target_servers=None)
