class NumOfTargetServersInsufficientError(Exception):
    def __init__(
        self,
        msg: str,
    ):
        prefix: str = (
            "The current analysis supports only simples cases where there"
            " are no waits on target servers. For every k drafts that are"
            " ready for verification, there must be an idle target server. "
        )
        super().__init__(prefix + msg)


class DrafterSlowerThanTargetError(Exception):
    def __init__(
        self,
        msg: str,
    ):
        prefix: str = (
            "The drafter must be as fast as the target or faster than the target. "
        )
        super().__init__(prefix + msg)


class MissingHeatmapColumnError(Exception):
    def __init__(self, column: str):
        msg: str = (
            "Missing required column in DataFrame."
            f" Column '{column}' is required for the operation."
        )
        super().__init__(msg)


class IncompatibleExtendError(Exception):
    def __init__(self, source_type: str, target_type: str):
        prefix: str = "Attempt to append data from incompatible types. "
        msg: str = (
            f"{prefix}Append operation requires both objects to be of the same type. "
            f"Source type: {source_type}, Target type: {target_type}."
        )
        super().__init__(msg)


class HeatmapConfigInvalidAcceptanceRateRangeError(Exception):
    def __init__(self, msg: str):
        prefix: str = "Invalid acceptance rate range. "
        super().__init__(prefix + msg)


class ConfigPlotHeatmapInvalidLevelsStepError(Exception):
    def __init__(self, levels_step: float):
        msg: str = (
            "Invalid levels step. Levels step must be a factor of 1."
            f" Received levels step: {levels_step}, so that {1/levels_step=}"
            " is not an integer."
        )
        super().__init__(msg)


class InvalidGenConfigError(Exception):
    def __init__(
        self,
        msg: str,
    ):
        prefix: str = "The generation arguments must be compatible with each other. "
        super().__init__(prefix + msg)


class InvalidHeatmapKeyError(Exception):
    def __init__(self, key):
        msg = f"Invalid key '{key}'. Key must be an instance of HeatmapColumn enum."
        super().__init__(msg)


class DatasetMismatchError(Exception):
    def __init__(self, target_dataset: str, drafter_dataset: str):
        msg: str = (
            f"Dataset mismatch between target and drafter configurations. "
            f"Target dataset: {target_dataset}, Drafter dataset: {drafter_dataset}."
        )
        super().__init__(msg)


class UnsupportedDatasetError(Exception):
    def __init__(self, dataset: str, supported_datasets: list[str]):
        msg: str = (
            f"Unsupported dataset: {dataset}. "
            f"Supported datasets: {', '.join(supported_datasets)}."
        )
        super().__init__(msg)
