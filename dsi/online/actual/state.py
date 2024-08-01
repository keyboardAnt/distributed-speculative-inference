class State:
    def __init__(self, initial_prompt: list[int]):
        self._tok_ids: list[int] = initial_prompt  # Initial prompt tokens, protected
        self._v: int = -1  # Index of the last verified token, protected

    def _rollback(self, i: int) -> None:
        self._tok_ids = self._tok_ids[: i + 1]
        self._v = i

    def update(self, i: int, tok_id: int) -> None:
        self._rollback(i - 1)
        self._tok_ids.append(tok_id)
        self._v = i

    def extend(self, tok_ids: list[int]) -> None:
        self._tok_ids.extend(tok_ids)
