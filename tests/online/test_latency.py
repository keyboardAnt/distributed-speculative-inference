from unittest.mock import MagicMock

import pytest
import torch
from pydantic import ValidationError

from dsi.configs.experiment.latency import DTYPE_MAP, ConfigLatency
from dsi.online.latency.dataset import Dataset
from dsi.online.latency.experiment import ExperimentLatency
from dsi.online.latency.prompts import get_prompt


def test_config_latency_default_initialization():
    config = ConfigLatency(model="m", dataset="d")
    assert config.model == "m"
    assert config.dtype == "bfloat16"
    assert config.dataset == "d"
    assert config.num_examples == 50
    assert config.max_new_tokens == 20
    assert config.compile_model is False
    assert config.revision is None
    assert config.subset is None
    assert config.split == "test"


@pytest.mark.parametrize(
    "model,dtype,dataset,num_examples,max_new_tokens,compile_model,revision,subset,split",
    [
        (
            "model1",
            "torch.float32",
            "dataset1",
            100,
            30,
            True,
            "rev1",
            "subset1",
            "train",
        ),
        (
            "model2",
            "torch.float16",
            "dataset2",
            200,
            40,
            False,
            None,
            None,
            "validation",
        ),
    ],
)
def test_config_latency_custom_initialization(
    model,
    dtype,
    dataset,
    num_examples,
    max_new_tokens,
    compile_model,
    revision,
    subset,
    split,
):
    config = ConfigLatency(
        model=model,
        dtype=dtype,
        dataset=dataset,
        num_examples=num_examples,
        max_new_tokens=max_new_tokens,
        compile_model=compile_model,
        revision=revision,
        subset=subset,
        split=split,
    )
    assert config.model == model
    assert config.dtype == dtype
    assert config.dataset == dataset
    assert config.num_examples == num_examples
    assert config.max_new_tokens == max_new_tokens
    assert config.compile_model == compile_model
    assert config.revision == revision
    assert config.subset == subset
    assert config.split == split


@pytest.mark.parametrize("dtype,expected", DTYPE_MAP.items())
def test_get_torch_dtype_valid(dtype, expected):
    config = ConfigLatency(dtype=dtype)
    assert config.get_torch_dtype() == expected


def test_get_torch_dtype_invalid():
    config = ConfigLatency(dtype="invalid_dtype")
    assert config.get_torch_dtype() == torch.bfloat16


def test_config_latency_field_validation():
    with pytest.raises(ValidationError):
        ConfigLatency(num_examples="not_an_int")

    with pytest.raises(ValidationError):
        ConfigLatency(num_examples=-1)

    with pytest.raises(ValidationError):
        ConfigLatency(max_new_tokens=0)


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


# Mocking ConfigLatency and ConfigGen
@pytest.fixture
def mock_config_latency():
    return MagicMock(name="ConfigLatency")


@pytest.fixture
def mock_config_gen():
    return MagicMock(name="ConfigGen")


def test_experiment_latency_init_valid_configs(mock_config_latency, mock_config_gen):
    experiment = ExperimentLatency(
        config=mock_config_latency, gen_config=mock_config_gen
    )
    assert experiment.config == mock_config_latency
    assert experiment.gen_config == mock_config_gen


def test_experiment_latency_init_none_configs():
    with pytest.raises(TypeError):  # Assuming TypeError is raised for None configs
        ExperimentLatency(config=None, gen_config=None)


def test_experiment_latency_init_invalid_configs():
    with pytest.raises(TypeError):  # Assuming TypeError is raised for invalid configs
        ExperimentLatency(config="invalid_config", gen_config=123)
