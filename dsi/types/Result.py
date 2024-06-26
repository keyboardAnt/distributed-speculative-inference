from dataclasses import dataclass, field


@dataclass
class Result:
    """
    Args:
        cost_per_run: The total latency for each run
    """

    cost_per_run: list[float] = field(default_factory=list)
