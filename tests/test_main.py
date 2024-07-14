from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dsi.configs.cli import ConfigCLI, RunType
from dsi.main import offline_heatmap
from dsi.types.df_heatmap import DataFrameHeatmap


@pytest.fixture
def cfg():
    return ConfigCLI(load_csv=None, type=RunType.offline_heatmap)


@pytest.fixture
def mock_ray_manager():
    with patch("dsi.main.RayManager") as mock:
        yield mock


@pytest.fixture
def mock_enrich_inplace():
    with patch("dsi.main.enrich") as mock:
        yield mock


@pytest.fixture
def mock_dataframe_heatmap():
    with patch("dsi.main.DataFrameHeatmap") as mock:
        df_mock = pd.DataFrame(
            {
                "c": np.array([0.1, 0.2, 0.3]),
                "a": np.array([0.1, 0.2, 0.3]),
                "min_speedup_fed_vs_spec": np.array([1.1, 1.2, 1.3]),
            }
        )
        mock_instance = DataFrameHeatmap(df_mock)
        mock.from_heatmap_csv.return_value = mock_instance
        mock_instance.describe = lambda: "This is a mock of the `describe` method."
        yield mock


@pytest.fixture
def mock_logger():
    with patch("dsi.main.log") as mock:
        yield mock


@pytest.fixture
def mock_hydra_config():
    with patch(
        "hydra.core.hydra_config.HydraConfig.get",
        return_value=MagicMock(runtime=MagicMock(output_dir="mocked_output_dir")),
    ) as mock:
        yield mock


def test_offline_heatmap_new_experiment(
    cfg,
    mock_ray_manager,
    mock_enrich_inplace,
    mock_dataframe_heatmap,
    mock_hydra_config,
):
    cfg.load_csv = None
    # Create MagicMock objects for fig and ax
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    # Ensure the ax mock has the pcolormesh method
    mock_ax.pcolormesh = MagicMock()
    with patch(
        "matplotlib.pyplot.subplots", return_value=(mock_fig, mock_ax)
    ) as mock_subplots:
        offline_heatmap(cfg)
        # Assertions to ensure the subplots are correctly used and pcolormesh is called
        mock_subplots.assert_called()
    mock_ray_manager.assert_called_once_with(cfg.heatmap)
    mock_ray_manager.return_value.run.assert_called_once()
    mock_enrich_inplace.assert_called_once()
    mock_enrich_inplace.return_value.store.assert_called_once()
    mock_dataframe_heatmap.from_heatmap_csv.assert_not_called()


def test_offline_heatmap_load_existing(
    cfg, mock_ray_manager, mock_enrich_inplace, mock_dataframe_heatmap
):
    cfg.load_csv = "path/to/existing/heatmap.csv"
    try:
        offline_heatmap(cfg)
    except KeyError as e:
        print("Cannot plot a mocked DataFrame.")
        print(e)
    mock_ray_manager.assert_not_called()
    mock_enrich_inplace.assert_not_called()
    mock_dataframe_heatmap.from_heatmap_csv.assert_called_once()
