import copy

from dsi.online.actual.message import MsgVerifiedRightmost


class State:
    def __init__(self, initial_prompt: list[int]):
        self._tok_ids: list[int] = initial_prompt[:]  # Initial prompt tokens, protected
        self._v: int = (
            len(initial_prompt) - 1
        )  # Index of the last verified token, protected

    @property
    def v(self) -> int:
        return self._v

    @v.setter
    def v(self, v_new: int) -> None:
        self._v = v_new

    def extend(self, tok_ids: list[int], verified: bool) -> None:
        self._tok_ids.extend(tok_ids)
        if verified:
            self.v += len(tok_ids)

    def is_aligned(self, m: MsgVerifiedRightmost) -> bool:
        return self._tok_ids[m.v] == m.tok_id

    def rollback(self, i: int) -> None:
        self._tok_ids = self._tok_ids[: i + 1]
        self.v = i

    def clone(self) -> "State":
        """Returns a deep copy of the state."""
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"State(v={self.v}, tok_ids={self._tok_ids})"
