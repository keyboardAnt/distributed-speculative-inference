from unittest.mock import MagicMock, patch

import pytest
import torch

from dsi.configs.experiment.latency import ConfigLatency
from dsi.online.latency.dataset import Dataset
from dsi.online.latency.experiment import ExperimentLatency
from dsi.online.latency.prompts import get_prompt
from dsi.types.result import ResultLatency


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


@pytest.fixture
def config():
    return ConfigLatency(
        model="double7/vicuna-68m",
        dataset=Dataset.ALPACA,
        compile_model=False,
        num_ex=5,
        max_new_tokens=5,
        random_seed=42,
    )


@pytest.fixture
def model_mock():
    model = MagicMock()
    model.generate = MagicMock(return_value=MagicMock(shape=(1, 51)))
    return model


@pytest.fixture
def tokenizer_mock():
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 0
    return tokenizer


@patch("dsi.online.latency.experiment.load_dataset")
@patch("dsi.online.latency.experiment.AutoModelForCausalLM.from_pretrained")
@patch("dsi.online.latency.experiment.AutoTokenizer.from_pretrained")
@patch("dsi.online.latency.experiment.torch.compile")
def test_single_repeat_without_compile_model(
    torch_compile_mock,
    tokenizer_from_pretrained_mock,
    model_from_pretrained_mock,
    load_dataset_mock,
    config,
    model_mock,
    tokenizer_mock,
):
    load_dataset_mock.return_value.shuffle.return_value.select.return_value = [
        {"input": "test", "instruction": "summarize this input"}
    ] * config.num_examples
    model_from_pretrained_mock.return_value = model_mock
    tokenizer_from_pretrained_mock.return_value = tokenizer_mock

    experiment = ExperimentLatency(config)
    result = experiment._single_repeat()

    load_dataset_mock.assert_called_with(path=config.dataset, name=None, split="test")
    model_from_pretrained_mock.assert_called_with(
        config.model, device_map="auto", torch_dtype=torch.bfloat16, revision=None
    )
    tokenizer_from_pretrained_mock.assert_called_with(config.model)
    torch_compile_mock.assert_not_called()
    assert isinstance(result, ResultLatency)


@patch("dsi.online.latency.experiment.load_dataset")
@patch("dsi.online.latency.experiment.AutoModelForCausalLM.from_pretrained")
@patch("dsi.online.latency.experiment.AutoTokenizer.from_pretrained")
@patch("dsi.online.latency.experiment.torch.compile")
def test_single_repeat_with_compile_model(
    torch_compile_mock,
    tokenizer_from_pretrained_mock,
    model_from_pretrained_mock,
    load_dataset_mock,
    config,
    model_mock,
    tokenizer_mock,
):
    config.compile_model = True
    load_dataset_mock.return_value.shuffle.return_value.select.return_value = [
        {"input": "test", "instruction": "summarize this input"}
    ] * config.num_examples
    model_from_pretrained_mock.return_value = model_mock
    tokenizer_from_pretrained_mock.return_value = tokenizer_mock

    experiment = ExperimentLatency(config)
    result = experiment._single_repeat()

    torch_compile_mock.assert_called_once_with(model_mock)
    assert isinstance(result, ResultLatency)
