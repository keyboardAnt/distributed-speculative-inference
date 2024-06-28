"""Defines types for the results of the simulations."""

from dataclasses import dataclass, field


@dataclass
class Result:
    """
    Args:
        cost_per_run: The total latency for each run
    """

    cost_per_run: list[float] = field(default_factory=list)


@dataclass
class ResultSI(Result):
    """
    In addition to the base result:
    Args:
        num_iters_per_run: The number of iterations for each run
    """

    num_iters_per_run: list[int] = field(default_factory=list)


@dataclass
class ResultSIVerbose(Result):
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
