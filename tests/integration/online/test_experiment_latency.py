from dsi.configs.experiment.latency import ConfigLatency
from dsi.online.latency.experiment import ExperimentLatency


def test_experiment():
    """A smoke test for the ExperimentLatency class."""
    ExperimentLatency(ConfigLatency())
