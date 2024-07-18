from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dsi.configs.experiment.simul.heatmap import ConfigHeatmap
from dsi.configs.experiment.simul.offline import ConfigDSI
from dsi.offline.heatmap.manager import Manager


@pytest.fixture
def sample_heatmap_config():
    return {
        "param1": "value1",
        "param2": "value2",
    }


@pytest.fixture
def sample_dsi_config():
    return {"model_copy": MagicMock(return_value=MagicMock())}


@pytest.fixture
def manager(sample_heatmap_config, sample_dsi_config):
    heatmap_config = ConfigHeatmap(**sample_heatmap_config)
    dsi_config = ConfigDSI(**sample_dsi_config)
    return Manager(config_heatmap=heatmap_config, simul_defaults=dsi_config)


def test_initialization(manager):
    assert isinstance(manager._df_config_heatmap, pd.DataFrame)
    assert manager._results_raw is None


def test_update_config_simul(manager):
    config_simul = ConfigDSI()
    row = pd.Series({"c": 1, "a": 2, "k": 3, "num_target_servers": 4})
    updated_config = manager._update_config_simul(config_simul, row)
    assert updated_config.c == 1
    assert updated_config.a == 2
    assert updated_config.k == 3
    assert updated_config.num_target_servers == 4


def test_merge_results(manager):
    manager._results_raw = [(0, {"result_key": 123})]
    manager.df_results = pd.DataFrame(index=[0], columns=["result_key"])
    manager._merge_results()
    assert manager.df_results.loc[0, "result_key"] == 123


@pytest.fixture
def mock_ray(mocker):
    ray = mocker.MagicMock()
    with patch("dsi.types.heatmap.manager.ray", ray):
        yield ray


def test_run(mock_ray, manager):
    # Configure your mocks
    mock_ray.remote.return_value = MagicMock(remote=MagicMock())
    mock_ray.get.return_value = [(0, {"result_key": "result_value"})]

    # Execute
    result_df = manager.run()

    # Asserts
    mock_ray.init.assert_called_once()
    assert not result_df.empty
    assert "result_key" in result_df.columns
