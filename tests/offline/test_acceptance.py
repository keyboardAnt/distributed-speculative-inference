from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import AutoTokenizer

from dsi.configs.experiment.acceptance import ConfigAcceptanteRate
from dsi.configs.experiment.generation import ConfigGen
from dsi.offline.acceptance.experiment import ExperimentAcceptanceRate
from dsi.types.result import ResultAcceptance


@pytest.fixture
def experiment():
    config = ConfigAcceptanteRate(
        model="target_model", dataset="dataset", draft_model="draft_model"
    )
    gen_config = ConfigGen()
    draft_gen_config = ConfigGen()
    return ExperimentAcceptanceRate(config, gen_config, draft_gen_config)


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
        mock_model.return_value.device = torch.device("cpu")
        mock_tokenizer.return_value = MagicMock()
        mock_examples.return_value = ["example1"]
        yield mock_model, mock_tokenizer, mock_examples


def test_are_tokenizers_same_identical(experiment):
    tokenizer1 = AutoTokenizer.from_pretrained("double7/vicuna-68m")
    tokenizer2 = AutoTokenizer.from_pretrained("double7/vicuna-68m")
    assert experiment._are_tokenizers_same(tokenizer1, tokenizer2)


def test_are_tokenizers_same_diff_config(experiment):
    tokenizer1 = MagicMock()
    tokenizer2 = MagicMock()
    tokenizer1.config = {"model_type": "bigcode/starcoder"}
    tokenizer2.config = {"model_type": "double7/vicuna-68m"}
    assert not experiment._are_tokenizers_same(tokenizer1, tokenizer2)


def test_are_tokenizers_same_diff_vocab(experiment):
    tokenizer1 = MagicMock()
    tokenizer2 = MagicMock()
    tokenizer1.get_vocab.return_value = {"hello": 1, "world": 2}
    tokenizer2.get_vocab.return_value = {"hello": 1, "python": 3}
    assert not experiment._are_tokenizers_same(tokenizer1, tokenizer2)


def test_are_tokenizers_same_diff_special_tokens(experiment):
    tokenizer1 = MagicMock()
    tokenizer2 = MagicMock()
    tokenizer1.eos_token_id = 1
    tokenizer2.eos_token_id = 2
    tokenizer1.pad_token_id = 0
    tokenizer2.pad_token_id = 0
    tokenizer1.bos_token_id = -1
    tokenizer2.bos_token_id = -1
    tokenizer1.unk_token_id = 3
    tokenizer2.unk_token_id = 3
    assert not experiment._are_tokenizers_same(tokenizer1, tokenizer2)


def test_single_repeat_all_match(experiment, mock_dependencies):
    mock_model, _, _ = mock_dependencies
    # Mock the generate method to simulate target and draft models
    # producing the same output
    mock_model.return_value.generate.side_effect = [
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[0]]),
        torch.tensor([[0, 1]]),
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[0, 1, 2, 3]]),
    ]
    result = experiment._single_repeat()
    assert isinstance(result, ResultAcceptance)
    # Since all tokens match, acceptance rate should be 1
    assert result.acceptance_rate[0] == 0.8


def test_single_repeat_no_match(experiment, mock_dependencies):
    mock_model, _, _ = mock_dependencies
    # Mock the generate method to simulate target and draft models
    # producing different outputs
    mock_model.return_value.generate.side_effect = [
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[4]]),
        torch.tensor([[0, 5]]),
        torch.tensor([[0, 1, 6]]),
        torch.tensor([[0, 1, 2, 7]]),
    ]
    result = experiment._single_repeat()
    assert isinstance(result, ResultAcceptance)
    # Since no tokens match, acceptance rate should be 0
    assert result.acceptance_rate[0] == 0


def test_single_repeat_partial_match(experiment, mock_dependencies):
    mock_model, _, _ = mock_dependencies
    # Mock the generate method to simulate target and draft models
    # producing partially matching outputs
    mock_model.return_value.generate.side_effect = [
        torch.tensor([[0, 1, 2, 3]]),
        torch.tensor([[0]]),
        torch.tensor([[0, 4]]),
        torch.tensor([[0, 1, 2]]),
        torch.tensor([[0, 1, 2, 5]]),
    ]
    result = experiment._single_repeat()
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
