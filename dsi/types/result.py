"""Defines types for the results of the simulations."""

from dataclasses import dataclass, field, fields

from dsi.types.exception import IncompatibleAppendError


@dataclass
class _Result:
    """
    An abstract class for the results of a simulation.
    """

    pass

    def extend(self, to_append: "ResultSimul"):
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
class ResultSimul(_Result):
    """
    Args:
        cost_per_repeat: The total latency for each repeat
        num_iters_per_repeat: The number of iterations for each repeat
    """

    cost_per_repeat: list[float] = field(default_factory=list)
    num_iters_per_repeat: list[int] = field(default_factory=list)
