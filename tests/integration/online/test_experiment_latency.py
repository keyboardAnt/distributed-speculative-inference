from itertools import product

import pytest

from dsi.configs.experiment.latency import ConfigLatency
from dsi.online.latency.experiment import ExperimentLatency
from dsi.types.result import ResultLatency

models = ["double7/vicuna-68m", "gpt2"]
datasets = [
    {"dataset": "glue", "subset": "mrpc", "split": "train"},
    {"dataset": "openai_humaneval"},
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


@pytest.mark.skip(reason="#40")
def test_experiment(config_latency: ConfigLatency):
    """A smoke test for the ExperimentLatency class."""
    e = ExperimentLatency(config_latency)
    res: ResultLatency = e.run()
    assert all([latency > 0 for latency in res.ttft])
    assert all([latency > 0 for latency in res.tpot])
    assert len(res.ttft) == len(res.tpot)
