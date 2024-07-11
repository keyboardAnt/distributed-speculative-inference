import pytest

from dsi.configs.experiment.generation import ConfigGen
from dsi.types.exception import InvalidGenConfigError

def test_valid_config_sampling_enabled():
    config = ConfigGen(do_sample=True, temperature=0.5, top_p=0.9)
    config.model_post_init(None)
    assert config.temperature == 0.5
    assert config.top_p == 0.9

def test_valid_config_sampling_disabled():
    config = ConfigGen(do_sample=False, temperature=1.0, top_p=1.0)
    config.model_post_init(None)
    assert config.temperature == 1.0
    assert config.top_p == 1.0

def test_invalid_config_sampling_enabled_temperature_zero():
    with pytest.raises(InvalidGenConfigError, match="temperature must be different than 0 when do_sample is True."):
        config = ConfigGen(do_sample=True, temperature=0, top_p=0.9)
        config.model_post_init(None)

def test_invalid_config_sampling_disabled_temperature_not_one():
    with pytest.raises(InvalidGenConfigError, match="temperature must be 1.0 when do_sample is False."):
        config = ConfigGen(do_sample=False, temperature=0.5, top_p=1.0)
        config.model_post_init(None)

def test_invalid_config_sampling_disabled_top_p_not_one():
    with pytest.raises(InvalidGenConfigError, match="top_p must be 1.0 when do_sample is False."):
        config = ConfigGen(do_sample=False, temperature=1.0, top_p=0.9)
        config.model_post_init(None)

def test_edge_case_temperature_close_to_zero_sampling_enabled():
    config = ConfigGen(do_sample=True, temperature=0.0001, top_p=0.9)
    config.model_post_init(None)
    assert config.temperature == 0.0001

def test_edge_case_temperature_and_top_p_one_sampling_disabled():
    config = ConfigGen(do_sample=False, temperature=1.0, top_p=1.0)
    config.model_post_init(None)
    assert config.temperature == 1.0
    assert config.top_p == 1.0