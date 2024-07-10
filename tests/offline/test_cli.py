from unittest.mock import patch

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
    with patch("dsi.main.enrich_inplace") as mock:
        yield mock


@pytest.fixture
def mock_dataframe_heatmap():
    with patch("dsi.main.DataFrameHeatmap") as mock:
        mock_instance = DataFrameHeatmap(pd.DataFrame())
        mock.from_heatmap_csv.return_value = mock_instance
        mock_instance.describe = lambda: "This is a mock of the `describe` method."
        yield mock


@pytest.fixture
def mock_logger():
    with patch("dsi.main.log") as mock:
        yield mock


# Corrected test function with appropriate fixture names and usage
def test_offline_heatmap_new_experiment(
    cfg, mock_ray_manager, mock_enrich_inplace, mock_dataframe_heatmap, mock_logger
):
    cfg.load_csv = None
    offline_heatmap(cfg)
    mock_ray_manager.assert_called_once_with(cfg.heatmap)
    mock_ray_manager.return_value.run.assert_called_once()
    mock_enrich_inplace.assert_called_once()
    mock_enrich_inplace.return_value.store.assert_called_once()
    mock_dataframe_heatmap.from_heatmap_csv.assert_not_called()


def test_offline_heatmap_load_existing(cfg, mock_dataframe_heatmap, mock_logger):
    cfg.load_csv = "path/to/existing/heatmap.csv"
    offline_heatmap(cfg)
    mock_dataframe_heatmap.from_heatmap_csv.assert_called_once_with(
        "path/to/existing/heatmap.csv"
    )
    mock_logger.info.assert_any_call(
        "Loading results from %s", "path/to/existing/heatmap.csv"
    )
