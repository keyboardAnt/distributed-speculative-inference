import multiprocessing
from contextlib import suppress

from dsi.online.actual.message import MsgVerifiedRightmost


class InvalidRollbackError(Exception):
    def __init__(
        self,
        i: int,
    ):
        super().__init__(f"Rollback index {i} is out of bounds.")


class State:
    def __init__(self, initial_prompt: list[int]):
        self._tok_ids: list[int] = initial_prompt[:]
        self._v: int = len(initial_prompt) - 1
        self._lock = multiprocessing.Lock()

    @property
    def tok_ids(self) -> list[int]:
        with self._lock:
            return self._tok_ids

    @tok_ids.setter
    def tok_ids(self, tok_ids_new: list[int]) -> None:
        with self._lock:
            self._tok_ids = tok_ids_new

    @property
    def v(self) -> int:
        with self._lock:
            return self._v

    @v.setter
    def v(self, v_new: int) -> None:
        with self._lock:
            self._v = v_new

    def extend(self, tok_ids: list[int], verified: bool) -> None:
        self.tok_ids.extend(tok_ids)
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
        ret = (
            State(self.tok_ids[: self.v + 1]) if only_verified else State(self.tok_ids)
        )
        ret.v = self.v
        return ret

    def to_dict(self) -> dict[str, int]:
        """Serialize the state to a dictionary."""
        return {"tok_ids": self.tok_ids, "v": self.v}

    @classmethod
    def from_dict(cls, data: dict) -> "State":
        """Deserialize the dictionary back to a State object."""
        instance = cls([])
        instance.tok_ids = data["tok_ids"]
        instance._v = data["v"]
        return instance

    def __repr__(self) -> str:
        return f"State(v={self.v}, tok_ids={self.tok_ids})"
