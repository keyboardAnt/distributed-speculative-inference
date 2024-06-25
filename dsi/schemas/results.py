"""Defines schemas for the results of the simulations."""

from dataclasses import dataclass, field

from dsi.schemas.run import Result


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
    """

    num_toks_of_last_iter_per_run: list[int] = field(default_factory=list)
    num_toks_per_iter: list[int] = field(default_factory=list)
