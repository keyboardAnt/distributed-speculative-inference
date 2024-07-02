import pandas as pd

from dsi.types.name import HeatmapColumn, Param


class DataFrameHeatmap(pd.DataFrame):
    @classmethod
    def from_heatmap_csv(cls, filepath):
        """
        Factory method to create HeatmapDataFrame from a CSV file.
        Includes validation of column names against Heatmap-specific requirements.
        """
        data = pd.read_csv(filepath)
        return cls(data).validate_columns()

    @classmethod
    def from_dataframe(cls, data):
        """
        Factory method to create HeatmapDataFrame from an existing DataFrame.
        Includes validation of column names against Heatmap-specific requirements.
        """
        return cls(data).validate_columns()

    def validate_columns(self):
        """
        Validates the column names to ensure they match expected HeatmapColumn or Param
        names. Raises ValueError if any column names are invalid.
        """
        valid_columns = (
            HeatmapColumn.get_all_valid_values() + Param.get_all_valid_values()
        )
        for column in self.columns:
            if column not in valid_columns:
                raise ValueError(
                    f"Invalid column name: {column}. Column names must be one of the"
                    " defined HeatmapColumn or Param names."
                )
        return self
