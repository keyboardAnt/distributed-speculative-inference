from contextlib import suppress
from multiprocessing import Lock, Manager

from dsi.online.actual.message import MsgVerifiedRightmost


class InvalidRollbackError(Exception):
    def __init__(
        self,
        i: int,
    ):
        super().__init__(f"Rollback index {i} is out of bounds.")


class State:
    def __init__(self, initial_prompt: list[int]):
        manager = Manager()
        self._tok_ids: list[int] = manager.list(initial_prompt[:])
        self._v = manager.Value("i", len(initial_prompt) - 1)
        self._lock = Lock()

    @property
    def tok_ids(self) -> list[int]:
        with self._lock:
            return list(self._tok_ids)

    @tok_ids.setter
    def tok_ids(self, tok_ids_new: list[int]) -> None:
        with self._lock:
            self._tok_ids[:] = tok_ids_new

    @property
    def v(self) -> int:
        with self._lock:
            return self._v.value

    @v.setter
    def v(self, v_new: int) -> None:
        with self._lock:
            self._v.value = v_new

    def extend(self, tok_ids: list[int], verified: bool) -> None:
        self.tok_ids += tok_ids
        if verified:
            self.v += len(tok_ids)

    def is_aligned(self, m: MsgVerifiedRightmost) -> bool:
        """
        Check if the state is aligned with the given message.
        """
        with suppress(IndexError):
            return self.tok_ids[m.v] == m.tok_id
        return False

    def rollback(self, i: int) -> None:
        """
        Rollback the state so that the last verified token is at index `i`.
        Raises an InvalidRollbackError if `i` is out of bounds.
        """
        if i >= len(self.tok_ids):
            raise InvalidRollbackError(i)
        self.tok_ids = self.tok_ids[: i + 1]
        self.v = i

    def clone(self, only_verified: bool) -> "State":
        """Returns a deep copy of the state."""
        if only_verified:
            return State(self.tok_ids[: self.v + 1])
        ret = State(self.tok_ids)
        ret.v = self.v
        return ret

    def __repr__(self) -> str:
        return f"State(v={self.v}, tok_ids={self.tok_ids})"
