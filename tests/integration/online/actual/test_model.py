from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import GPT2LMHeadModel

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.model import Model, SetupModel
from dsi.online.actual.state import State


@pytest.fixture
def mock_state():
    state = State(tok_ids=[1, 2, 3])
    state.extend = MagicMock()
    state.rollback = MagicMock()
    return state


@pytest.fixture
def mock_model(mock_state):
    with patch("torch.cuda.is_available", return_value=True):
        model = Model(SetupModel(gpu_id=0, state=mock_state, _name="gpt2"))
    model._model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4, 5]]))
    return model


def test_model_initialization(mock_model):
    assert mock_model.setup.gpu_id == 0
    assert isinstance(mock_model._model, GPT2LMHeadModel)
    assert isinstance(mock_model.setup.state, State)


# Test draft method with positive max_new_tokens
def test_draft_with_tokens_mocked(mock_model):
    mock_model.draft(max_new_tokens=5)
    mock_model._model.generate.assert_called_once()
    mock_model.setup.state.extend.assert_called()


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
            SetupModel(gpu_id=0, state=State([1, 2, 3]), _name="gpt2")
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
                SetupModel(gpu_id=0, state=State([1, 2, 3]), _name="nonexistent-model")
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
    model = Model(SetupModel(gpu_id=0, state=state, _name="gpt2"))
    expected_result = 2
    while expected_result >= 0:
        assert model._get_num_accepted(logits) == expected_result
        state.rollback(n - 1)
        tok_ids = tok_ids[1:]
        state.extend(tok_ids, verified=False)
        logits = logits[1:]
        expected_result -= 1


def test_get_num_accepted_all_match():
    state = State([0, 1])
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    tok_ids = [1, 0, 1]
    state.extend(tok_ids, verified=False)
    model = Model(SetupModel(gpu_id=0, state=state, _name="gpt2"))
    assert model._get_num_accepted(logits) == 3


def test_get_num_accepted_distinct_matching_substrings():
    state = State([0, 1])
    logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
    tok_ids = [1, 1, 1]
    state.extend(tok_ids, verified=False)
    model = Model(SetupModel(gpu_id=0, state=state, _name="gpt2"))
    assert model._get_num_accepted(logits) == 1


@pytest.fixture
def model() -> Model:
    # The prompt is the GPT2 encoding of "The president of USA is Barack Ob"
    state = State(tok_ids=[464, 1893, 286, 4916, 318, 8732, 1835])
    return Model(SetupModel(gpu_id=0, state=state, _name="gpt2"))


def test_draft(model):
    assert model.setup.state.v == 6
    result = model.draft(max_new_tokens=8)
    assert model.setup.state.v == 6
    assert result == [17485, 11, 508, 318, 262, 717, 5510, 12]


def test_verify_accept(model: Model):
    v: int = model.setup.state.v
    tok_ids = [17485, 11, 508, 318, 262, 717, 5510, 12]
    result: MsgVerifiedRightmost = model.verify(tok_ids)
    assert isinstance(result, MsgVerifiedRightmost)
    assert result.tok_id == model.setup.state.tok_ids[-1]
    assert model.setup.state.v == v + len(tok_ids) + 1


def test_verify_reject_rightmost(model: Model):
    v: int = model.setup.state.v
    tok_ids = [17485, 11, 508, 318, 262, 717, 5510, 11]
    result: MsgVerifiedRightmost = model.verify(tok_ids)
    assert isinstance(result, MsgVerifiedRightmost)
    expected_tok_id = 12
    assert result.tok_id == expected_tok_id
    assert model.setup.state.tok_ids[-1] == expected_tok_id
    assert model.setup.state.v == v + len(tok_ids)


def test_verify_reject(model: Model):
    v: int = model.setup.state.v
    tok_ids = [17485, 11, 508, 318, 262, 262, 5510, 12]
    result: MsgVerifiedRightmost = model.verify(tok_ids)
    assert isinstance(result, MsgVerifiedRightmost)
    expected_tok_id = 717
    assert result.tok_id == expected_tok_id
    assert model.setup.state.tok_ids[-1] == expected_tok_id
    assert model.setup.state.v == v + 6
