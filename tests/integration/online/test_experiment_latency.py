from itertools import product

import pytest

from dsi.configs.experiment.latency import ConfigLatency
from dsi.online.latency.experiment import ExperimentLatency
from dsi.types.result import ResultLatency

models = ["double7/vicuna-68m", "gpt2"]
datasets = [
    {"dataset": "glue", "subset": "mrpc", "split": "train"},
    {"dataset": "openai/openai_humaneval", "split": "test"},
]
combined_params = [
    {"model": model, **dataset} for model, dataset in product(models, datasets)
]


@pytest.fixture(params=combined_params)
def config_latency(request) -> ConfigLatency:
    return ConfigLatency(**request.param)


# @pytest.mark.skip(reason="#")
def test_experiment(config_latency: ConfigLatency):
    """A smoke test for the ExperimentLatency class."""
    e = ExperimentLatency(config_latency)
    res: ResultLatency = e.run()
    assert all([latency > 0 for latency in res.ttft])
    assert all([latency > 0 for latency in res.tpot])
    assert len(res.ttft) == len(res.tpot)
