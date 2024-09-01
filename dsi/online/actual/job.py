from dataclasses import dataclass


@dataclass
class Job:
    """
    Represents a verification job.
    Args:
        i: int - The index of the first draft token.
        tok_ids: list - The list of token ids to verify.
    """

    i: int
    tok_ids: list[int]
