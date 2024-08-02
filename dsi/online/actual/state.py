from threading import Lock

from dsi.online.actual.message import MsgVerifiedRightmost


class State:
    def __init__(self, initial_prompt: list[int]):
        self._lock = Lock()
        self._tok_ids: list[int] = initial_prompt  # Initial prompt tokens, protected
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
        with self._lock:
            self._tok_ids.extend(tok_ids)
            if verified:
                self.v += len(tok_ids)

    def is_aligned(self, m: MsgVerifiedRightmost) -> bool:
        return self._tok_ids[m.v] == m.tok_id

    def rollback(self, i: int) -> None:
        with self._lock:
            self._tok_ids = self._tok_ids[: i + 1]
            self.v = i

    def clone(self) -> "State":
        """Returns a deep copy of the state."""
        ret = State(self._tok_ids[:])
        ret.v = self._v
        return ret

    def to_dict(self) -> dict[str, int]:
        """Serialize the state to a dictionary."""
        return {"tok_ids": self._tok_ids, "v": self._v}

    @classmethod
    def from_dict(cls, data: dict) -> "State":
        """Deserialize the dictionary back to a State object."""
        instance = cls([])
        instance._tok_ids = data["tok_ids"]
        instance._v = data["v"]
        return instance

    def __repr__(self) -> str:
        return f"State(v={self.v}, tok_ids={self._tok_ids})"
