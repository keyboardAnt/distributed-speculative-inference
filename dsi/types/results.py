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
