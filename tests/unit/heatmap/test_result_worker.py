import pytest

from dsi.types.exception import InvalidHeatmapKeyError
from dsi.types.name import HeatmapColumn
from dsi.types.result import ResultWorker


def test_initialization_with_valid_keys():
    # Initialize ResultWorker with all valid keys and some dummy values
    valid_initialization = {
        HeatmapColumn.cost_si: 10,
        HeatmapColumn.cost_nonsi: 20,
        HeatmapColumn.cost_dsi: 30,
        HeatmapColumn.cost_baseline: 40,
    }
    worker = ResultWorker(**valid_initialization)
    # Check that all keys and values are correctly set
    for key, value in valid_initialization.items():
        assert worker[key] == value


def test_initialization_with_invalid_keys():
    # Attempt to initialize with an invalid key
    with pytest.raises(InvalidHeatmapKeyError):
        ResultWorker(invalid_key=100)


def test_setitem_with_valid_keys():
    # Create an instance and set an item with a valid key
    worker = ResultWorker()
    worker[HeatmapColumn.min_cost_si] = 50
    assert worker[HeatmapColumn.min_cost_si] == 50


def test_setitem_with_invalid_keys():
    # Attempt to set an item with an invalid key
    worker = ResultWorker()
    with pytest.raises(InvalidHeatmapKeyError):
        worker["invalid_key"] = 100


def test_all_valid_heatmap_keys():
    # Test setting all possible valid HeatmapColumn keys
    worker = ResultWorker()
    for key in HeatmapColumn.get_all_valid_values():
        try:
            worker[key] = HeatmapColumn[key]
            assert worker[key] == HeatmapColumn[key]
        except Exception as e:
            pytest.fail(
                f"Unexpected {e.__class__.__name__} exception with message: {e}"
            )
