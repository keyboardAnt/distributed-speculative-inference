from unittest.mock import MagicMock, patch

import pytest
import torch

from dsi.configs.experiment.latency import ConfigLatency
from dsi.online.latency.dataset import Dataset
from dsi.online.latency.experiment import ExperimentLatency
from dsi.online.latency.prompts import get_prompt


@pytest.fixture
def samsum_example():
    return {"dialogue": "Speaker 1: Hello\nSpeaker 2: Hi"}


@pytest.fixture
def cnn_dailymail_example():
    return {"article": "This is an article."}


@pytest.fixture
def mbpp_example():
    return {
        "text": "Write a Python function to get the unique elements of a list.",
        "test_list": ["assert get_unique_elements([1, 2, 3, 2, 1]) == [1, 2, 3]"],
    }


@pytest.fixture
def alpaca_example_with_input():
    return {"instruction": "Summarize the text", "input": "This is a text."}


@pytest.fixture
def alpaca_example_without_input():
    return {"instruction": "Summarize the text", "input": ""}


@pytest.fixture
def default_example():
    return {"prompt": "This is a default prompt."}


def test_get_prompt_samsum(samsum_example):
    prompt = get_prompt(Dataset.SAMSUM, samsum_example)
    expected = (
        "Summarize this dialog:\nSpeaker 1: Hello\nSpeaker 2: Hi\n---\nSummary:\n"
    )
    assert prompt == expected


def test_get_prompt_cnn_dailymail(cnn_dailymail_example):
    prompt = get_prompt(Dataset.CNN_DAILYMAIL, cnn_dailymail_example)
    expected = "Summarize:\nThis is an article.\nSummary:\n"
    assert prompt == expected


def test_get_prompt_mbpp(mbpp_example):
    prompt = get_prompt(Dataset.MBPP, mbpp_example)
    expected_start = (
        "[INST]Your task is to write a Python function to solve a programming problem."
    )
    assert prompt.startswith(expected_start)


def test_get_prompt_alpaca_with_input(alpaca_example_with_input):
    prompt = get_prompt(Dataset.ALPACA, alpaca_example_with_input)
    expected_start = (
        "Below is an instruction that describes a task,"
        " paired with an input that provides further context."
    )
    assert prompt.startswith(expected_start)


def test_get_prompt_alpaca_without_input(alpaca_example_without_input):
    prompt = get_prompt(Dataset.ALPACA, alpaca_example_without_input)
    expected_start = (
        "Below is an instruction that describes a task."
        " Write a response that appropriately completes the request."
    )
    assert prompt.startswith(expected_start)


@pytest.fixture(
    params=[
        {"compile_model": False, "num_ex": 10, "random_seed": 42, "max_new_tokens": 10},
        {"compile_model": True, "num_ex": 5, "random_seed": 24, "max_new_tokens": 20},
    ]
)
def config_latency(request):
    return ConfigLatency(
        dataset=Dataset.ALPACA, model="double7/vicuna-68m", **request.param
    )


@pytest.fixture
def experiment_latency(config_latency):
    return ExperimentLatency(config=config_latency)


@pytest.fixture
def model_tokenizer():
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    return model, tokenizer


@patch("dsi.online.latency.experiment.AutoModelForCausalLM.from_pretrained")
@patch("dsi.online.latency.experiment.AutoTokenizer.from_pretrained")
def test_load_model_tokenizer(
    mock_tokenizer, mock_model, experiment_latency, model_tokenizer
):
    mock_model.return_value = model_tokenizer[0]
    mock_tokenizer.return_value = model_tokenizer[1]
    model, tokenizer = experiment_latency._load_model_tokenizer("double7/vicuna-68m")
    mock_model.assert_called_once_with(
        "double7/vicuna-68m",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        revision=None,
    )
    mock_tokenizer.assert_called_once_with("double7/vicuna-68m")
    assert model is not None
    assert tokenizer is not None
