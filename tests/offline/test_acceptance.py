import pytest
import torch

from dsi.configs.experiment.acceptance import ConfigAcceptanteRate


def test_config_acceptance_initialization_defaults():
    config = ConfigAcceptanteRate(model="m", dataset="d", draft_model="dr")
    assert config.draft_model == "dr"
    assert config.draft_dtype == "bfloat16"
    assert config.draft_revision is None
    assert config.draft_compile_model is False


def test_config_acceptance_initialization_custom():
    config = ConfigAcceptanteRate(
        model="m",
        dataset="d",
        draft_model="test_model",
        draft_dtype="float32",
        draft_compile_model=True,
        draft_revision="test_revision",
    )
    assert config.draft_model == "test_model"
    assert config.draft_dtype == "float32"
    assert config.draft_compile_model is True
    assert config.draft_revision == "test_revision"


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_get_torch_dtype(dtype, expected):
    config = ConfigAcceptanteRate(
        model="m", dataset="d", draft_model="dr", draft_dtype=dtype
    )
    assert config.get_torch_draft_dtype() == expected


def test_draft_revision_optional():
    config = ConfigAcceptanteRate(model="m", dataset="d", draft_model="dr")
    assert config.draft_revision is None
    config = ConfigAcceptanteRate(
        model="m", dataset="d", draft_model="dr", draft_revision="rev1"
    )
    assert config.draft_revision == "rev1"
