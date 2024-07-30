from itertools import product

import pytest

from dsi.configs.experiment.latency import ConfigLatency
from dsi.online.latency.experiment import ExperimentLatency
from dsi.types.exception import UnsupportedDatasetError
from dsi.types.result import ResultLatency

models = ["double7/vicuna-68m", "bigcode/tiny_starcoder_py"]
datasets = [
    {"dataset": "cnn_dailymail", "subset": "2.0.0"},
    {"dataset": "openai/openai_humaneval"},
]


def generate_test_id(model, dataset):
    """Generate a readable test ID based on the model and dataset dictionary."""
    dataset_id = "_".join([f"{key}-{value}" for key, value in dataset.items()])
    return f"{model}_{dataset_id}"


# Combine models and datasets, and use a helper to create readable IDs
combined_params = list(product(models, datasets))
param_ids = [generate_test_id(model, dataset) for model, dataset in combined_params]


@pytest.fixture(params=combined_params, ids=param_ids)
def config_latency(request):
    model, dataset = request.param
    return ConfigLatency(model=model, num_examples=3, num_repeats=1, **dataset)


def test_experiment(config_latency: ConfigLatency):
    """A smoke test for the ExperimentLatency class."""
    e = ExperimentLatency(config_latency)
    res: ResultLatency = e.run()
    assert all([latency > 0 for latency in res.ttft])
    assert all([latency > 0 for latency in res.tpot])
    assert len(res.ttft) == len(res.tpot)


@pytest.fixture
def valid_config_latency():
    return dict(
        model="double7/vicuna-68m",
        dataset="cnn_dailymail",
        num_examples=3,
        num_repeats=1,
    )


@pytest.fixture
def invalid_config_latency():
    return dict(
        model="double7/vicuna-68m",
        dataset="unsupported_dataset",
        num_examples=3,
        num_repeats=1,
    )


def test_model_post_init_valid(valid_config_latency):
    try:
        ConfigLatency(**valid_config_latency)
    except UnsupportedDatasetError:
        pytest.fail("model_post_init raised UnsupportedDatasetError unexpectedly!")


def test_model_post_init_invalid(invalid_config_latency):
    with pytest.raises(UnsupportedDatasetError):
        ConfigLatency(**invalid_config_latency)
