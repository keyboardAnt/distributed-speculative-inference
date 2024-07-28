import pytest
import torch

from dsi.configs.experiment.latency import ConfigLatency


def test_config_latency_initialization_defaults():
    config = ConfigLatency(model="m", dataset="cnn_dailymail")
    assert config.model == "m"
    assert config.dtype == "bfloat16"
    assert config.dataset == "cnn_dailymail"
    assert config.num_examples == 50
    assert config.max_new_tokens == 20
    assert config.compile_model is False
    assert config.revision is None
    assert config.subset is None
    assert config.split == "test"


def test_config_latency_initialization_custom():
    config = ConfigLatency(
        model="test_model",
        dtype="float32",
        dataset="test_dataset",
        num_examples=100,
        max_new_tokens=40,
        compile_model=True,
        revision="test_revision",
        subset="test_subset",
        split="train",
    )
    assert config.model == "test_model"
    assert config.dtype == "float32"
    assert config.dataset == "test_dataset"
    assert config.num_examples == 100
    assert config.max_new_tokens == 40
    assert config.compile_model is True
    assert config.revision == "test_revision"
    assert config.subset == "test_subset"
    assert config.split == "train"


@pytest.mark.parametrize(
    "dtype,expected",
    [
        ("float32", torch.float32),
        ("float16", torch.float16),
        ("bfloat16", torch.bfloat16),
    ],
)
def test_get_torch_dtype(dtype, expected):
    config = ConfigLatency(model="m", dataset="cnn_dailymail", dtype=dtype)
    assert config.get_torch_dtype() == expected


def test_num_examples_constraint():
    with pytest.raises(ValueError):
        ConfigLatency(model="m", dataset="cnn_dailymail", num_examples=0)
    config = ConfigLatency(model="m", dataset="cnn_dailymail", num_examples=1)
    assert config.num_examples == 1


def test_max_new_tokens_constraint():
    with pytest.raises(ValueError):
        ConfigLatency(model="m", dataset="cnn_dailymail", max_new_tokens=0)
    config = ConfigLatency(model="m", dataset="cnn_dailymail", max_new_tokens=1)
    assert config.max_new_tokens == 1


def test_revision_and_subset_optional():
    config = ConfigLatency(
        model="m",
        dataset="cnn_dailymail",
    )
    assert config.revision is None
    assert config.subset is None
    config = ConfigLatency(
        model="m", dataset="cnn_dailymail", revision="rev1", subset="sub1"
    )
    assert config.revision == "rev1"
    assert config.subset == "sub1"


def test_split_validation():
    with pytest.raises(ValueError):
        ConfigLatency(model="m", dataset="cnn_dailymail", split=1)
    config = ConfigLatency(model="m", dataset="cnn_dailymail", split="test")
    assert config.split == "test"
