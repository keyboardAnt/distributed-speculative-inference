from unittest.mock import patch

import pandas as pd
import pytest

from dsi.types.df_heatmap import DataFrameHeatmap

# Mock valid column names for testing
valid_columns = ["col1", "col2", "param1", "param2"]


@pytest.fixture
def valid_dataframe():
    """Fixture to create a valid DataFrame for testing."""
    return pd.DataFrame(columns=valid_columns)


@pytest.fixture
def invalid_dataframe():
    """Fixture to create an invalid DataFrame for testing."""
    return pd.DataFrame(columns=["invalid_col", "another_invalid_col"])


@pytest.fixture
def valid_csv_file(tmp_path):
    """Fixture to create a valid CSV file for testing."""
    df = pd.DataFrame(columns=valid_columns)
    file_path = tmp_path / "valid.csv"
    df.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def invalid_csv_file(tmp_path):
    """Fixture to create an invalid CSV file for testing."""
    df = pd.DataFrame(columns=["invalid_col", "another_invalid_col"])
    file_path = tmp_path / "invalid.csv"
    df.to_csv(file_path, index=False)
    return file_path


@patch(
    "dsi.types.df_heatmap.HeatmapColumn.get_all_valid_values",
    return_value=valid_columns[:2],
)
@patch(
    "dsi.types.df_heatmap.Param.get_all_valid_values", return_value=valid_columns[2:]
)
def test_from_heatmap_csv_valid(mock_param, mock_heatmap, valid_csv_file):
    df = DataFrameHeatmap.from_heatmap_csv(valid_csv_file)
    assert isinstance(df, DataFrameHeatmap)


@patch(
    "dsi.types.df_heatmap.HeatmapColumn.get_all_valid_values",
    return_value=valid_columns[:2],
)
@patch(
    "dsi.types.df_heatmap.Param.get_all_valid_values", return_value=valid_columns[2:]
)
def test_from_heatmap_csv_invalid(mock_param, mock_heatmap, invalid_csv_file):
    with pytest.raises(ValueError):
        DataFrameHeatmap.from_heatmap_csv(invalid_csv_file)


@patch(
    "dsi.types.df_heatmap.HeatmapColumn.get_all_valid_values",
    return_value=valid_columns[:2],
)
@patch(
    "dsi.types.df_heatmap.Param.get_all_valid_values", return_value=valid_columns[2:]
)
def test_from_dataframe_valid(mock_param, mock_heatmap, valid_dataframe):
    df = DataFrameHeatmap.from_dataframe(valid_dataframe)
    assert isinstance(df, DataFrameHeatmap)


@patch(
    "dsi.types.df_heatmap.HeatmapColumn.get_all_valid_values",
    return_value=valid_columns[:2],
)
@patch(
    "dsi.types.df_heatmap.Param.get_all_valid_values", return_value=valid_columns[2:]
)
def test_from_dataframe_invalid(mock_param, mock_heatmap, invalid_dataframe):
    with pytest.raises(ValueError):
        DataFrameHeatmap.from_dataframe(invalid_dataframe)
