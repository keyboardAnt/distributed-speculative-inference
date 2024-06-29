"""Defines types for the results of the simulations."""

from dataclasses import dataclass, field, fields

from dsi.types.exception import IncompatibleAppendError


@dataclass
class Result:
    """
    Args:
        cost_per_run: The total latency for each run
        num_iters_per_run: The number of iterations for each run
    """

    cost_per_run: list[float] = field(default_factory=list)
    num_iters_per_run: list[int] = field(default_factory=list)

    def extend(self, to_append: "Result"):
        """
        Appends the values from another Result object to this one.

        Args:
            to_append (Result): The Result object from which to append data.
        """
        if not isinstance(to_append, type(self)):
            raise IncompatibleAppendError(type(self).__name__, type(to_append).__name__)

        for field_info in fields(self):
            # Check if both instances have the same field and it is a list
            current_list = getattr(self, field_info.name)
            if isinstance(current_list, list):
                appending_list = getattr(to_append, field_info.name, [])
                current_list.extend(appending_list)


@dataclass
class ResultVerbose(Result):
    """
    Extends ResultSI with the following:
    Args:
        num_toks_of_last_iter_per_run: The number of tokens in the last iteration for each run
        num_toks_per_iter: The number of tokens in each iteration, across all runs

    TODO(#13).
    """

    num_toks_of_last_iter_per_run: list[int] = field(default_factory=list)
    num_toks_per_iter: list[int] = field(default_factory=list)


class HeatmapColumn:
    cost_nonsi = "cost_nonspec"
    cost_si = "cost_spec"
    cost_dsi = "cost_fed"
    speedup_dsi_vs_si = "speedup_fed_vs_spec"
    speedup_dsi_vs_nonsi = "speedup_fed_vs_nonspec"
    speedup_si_vs_nonsi = "speedup_spec_vs_nonspec"
