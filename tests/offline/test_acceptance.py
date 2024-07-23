from unittest.mock import MagicMock, patch

import pytest
import torch

from dsi.configs.experiment.acceptance import ConfigAcceptanteRate
from dsi.configs.experiment.generation import ConfigGen
from dsi.offline.acceptance.experiment import ExperimentAcceptanceRate
from dsi.types.result import ResultAcceptance


@pytest.fixture
def setup_experiment():
    config = ConfigAcceptanteRate(
        model="target_model", dataset="dataset", draft_model="draft_model"
    )
    gen_config = ConfigGen()
    draft_gen_config = ConfigGen()
    experiment = ExperimentAcceptanceRate(config, gen_config, draft_gen_config)
    return experiment


@pytest.fixture
def mock_dependencies():
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained"
    ) as mock_model, patch(
        "transformers.AutoTokenizer.from_pretrained"
    ) as mock_tokenizer, patch.object(
        ExperimentAcceptanceRate, "_get_random_prompted_examples"
    ) as mock_examples:
        mock_model.return_value = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_examples.return_value = ["example1", "example2"]
        yield mock_model, mock_tokenizer, mock_examples


def test_single_repeat_all_match(setup_experiment, mock_dependencies):
    mock_model, _, _ = mock_dependencies
    # Mock the generate method to simulate target and draft models
    # producing the same output
    mock_model.return_value.generate.side_effect = lambda **kwargs: torch.tensor(
        [[0, 1, 2, 3]]
    )
    result = setup_experiment._single_repeat()
    assert isinstance(result, ResultAcceptance)
    # Since all tokens match, acceptance rate should be 1
    assert result.acceptance_rate[0] == 1


def test_single_repeat_no_match(setup_experiment, mock_dependencies):
    mock_model, _, _ = mock_dependencies
    # Mock the generate method to simulate target and draft models
    # producing different outputs
    mock_model.return_value.generate.side_effect = [
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[0, 4, 5, 6]]),
    ]
    result = setup_experiment._single_repeat()
    assert isinstance(result, ResultAcceptance)
    # Since no tokens match, acceptance rate should be 0
    assert result.acceptance_rate[0] == 0


def test_single_repeat_partial_match(setup_experiment, mock_dependencies):
    mock_model, _, _ = mock_dependencies
    # Mock the generate method to simulate target and draft models
    # producing partially matching outputs
    mock_model.return_value.generate.side_effect = [
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[0, 1, 5, 6]]),
    ]
    result = setup_experiment._single_repeat()
    assert isinstance(result, ResultAcceptance)
    # Since half of the tokens match, acceptance rate should be 0.5
    assert result.acceptance_rate[0] == 0.5


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
