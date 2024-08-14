from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import GPT2LMHeadModel

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.model import Model
from dsi.online.actual.state import State


@pytest.fixture
def mock_state():
    state = State(initial_prompt=[1, 2, 3])
    state.extend = MagicMock()
    state.rollback = MagicMock()
    return state


@pytest.fixture
def mock_model(mock_state):
    with patch("torch.cuda.is_available", return_value=True):
        model = Model(gpu_id=0, name="gpt2", is_verifier=True, state=mock_state)
    model._model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    return model


@pytest.fixture
def model() -> Model:
    # The prompt is the GPT2 encoding of "The president of USA is Barack Ob"
    state = State(initial_prompt=[464, 1893, 286, 4916, 318, 8732, 1835])
    return Model(gpu_id=0, name="gpt2", is_verifier=True, state=state)


def test_model_initialization(mock_model):
    assert mock_model.gpu_id == 0
    assert isinstance(mock_model._model, GPT2LMHeadModel)
    assert mock_model._is_verifier is True
    assert isinstance(mock_model.state, State)


# Test draft method with max_new_tokens <= 0
def test_draft_no_tokens(mock_model):
    result = mock_model.draft(max_new_tokens=0)
    assert result == []
    mock_model.state.extend.assert_not_called()


# Test draft method with positive max_new_tokens
def test_draft_with_tokens_mocked(mock_model):
    mock_model.draft(max_new_tokens=5)
    mock_model._model.generate.assert_called_once()
    mock_model.state.extend.assert_called()


@pytest.mark.skipif(torch.cuda.device_count() == 0, reason="Requires a GPU to run")
def test_model_device_assignment_cuda():
    model_with_gpu = Model(
        gpu_id=0, name="gpt2", is_verifier=True, state=State([1, 2, 3])
    )
    # Ensure the model is on the expected GPU
    assert model_with_gpu._model.device.type == "cuda"
    assert model_with_gpu._model.device.index == 0


def test_model_device_assignment_cpu():
    with patch("torch.cuda.device_count", return_value=0):  # No GPU available
        model_with_cpu = Model(
            gpu_id=0, name="gpt2", is_verifier=True, state=State([1, 2, 3])
        )
        # Ensure the model is on the CPU
        assert model_with_cpu._model.device.type == "cpu"


def test_model_loading_failure():
    with patch(
        "transformers.AutoModelForCausalLM.from_pretrained",
        side_effect=Exception("Failed to load model"),
    ):
        with pytest.raises(Exception) as excinfo:
            Model(
                gpu_id=0,
                name="nonexistent-model",
                is_verifier=True,
                state=State([1, 2, 3]),
            )
        assert "Failed to load model" in str(excinfo.value)


def test_get_num_accepted():
    # Setup actual model and state
    initial_prompt = [0, 1]
    n = len(initial_prompt)
    state = State(initial_prompt)
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    tok_ids = [1, 0, 0]
    state.extend(tok_ids, verified=False)
    model = Model(gpu_id=0, name="gpt2", is_verifier=True, state=state)
    result = model._get_num_accepted(logits)
    expected_result = 2
    assert result == expected_result
    while expected_result >= 0:
        state.rollback(n - 1)
        tok_ids = tok_ids[1:]
        state.extend(tok_ids, verified=False)
        logits = logits[1:]
        expected_result -= 1
        assert model._get_num_accepted(logits) == expected_result


def test_get_num_accepted_all_match():
    state = State([0, 1])
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    tok_ids = [1, 0, 1]
    state.extend(tok_ids, verified=False)
    model = Model(gpu_id=0, name="gpt2", is_verifier=True, state=state)
    assert model._get_num_accepted(logits) == 3


def test_get_num_accepted_distinct_matching_substrings():
    state = State([0, 1])
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    tok_ids = [1, 1, 1]
    state.extend(tok_ids, verified=False)
    model = Model(gpu_id=0, name="gpt2", is_verifier=True, state=state)
    assert model._get_num_accepted(logits) == 1


def test_verify_accept(model: Model):
    # Providing real input and performing verification
    tok_ids = [17485, 11, 508, 318, 262, 717, 5510, 12]
    v = model.state.v
    result: MsgVerifiedRightmost = model.verify(tok_ids)

    # Since _get_num_accepted and logits are calculated internally, no mocks are used
    assert isinstance(result, MsgVerifiedRightmost)
    assert (
        result.tok_id == model.state.tok_ids[-1]
    )  # Expects last token ID to be correct
    assert (
        model.state.v == v + len(tok_ids) - 1
    )  # This expects specific implementation details


# def test_verify_reject(model: Model):
#     # Providing real input and performing verification
#     tok_ids = [17485, 11, 508, 318, 262, 717, 5510, 11]
#     pass
