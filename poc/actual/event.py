import time
from dataclasses import dataclass
from uuid import UUID, uuid4
import torch


@dataclass
class Event:
    timestamp: float

    @classmethod
    def create(cls) -> "Event":
        return cls(timestamp=time.time())


@dataclass
class Message(Event):
    id: UUID

    @classmethod
    def create(cls) -> "Message":
        return cls(id=uuid4(), timestamp=time.time())


@dataclass
class Request(Message):
    """
    Args:
        tok_ids (torch.Tensor): The token IDs of the prompt. Shape: (seq_len,).
                                The prompt populates the first part of the sequence,
                                and the remaining positions are -1.
        n (int): The number of tokens to generate.
    """

    tok_ids: torch.Tensor
    n: int

    @classmethod
    def create(cls, tok_ids: torch.Tensor, n: int) -> "Request":
        return cls(id=uuid4(), timestamp=time.time(), tok_ids=tok_ids, n=n)

    def get_mask(self, seq_len: int, is_draft: bool) -> torch.Tensor:
        """
        Returns a boolean mask of shape (seq_len, ) where entries are True only at the
        positions that correspond to the response.
        If is_draft is True, the mask is True at n positions that follow the prompt (up
        to the end of the sequence).
        Otherwise, the mask is True at the n consecutive positions that end at the first
        -1 in tok_ids or the end of the sequence if there is no -1s.
        Examples:
        If tok_ids = [5, 3, 2, -1, -1], n = 2, and is_draft is True, then the mask is
        [False, False, False, True, True], and if is_draft is False, then the mask is
        [False, False, True, True, False].
        If tok_ids = [5, 3, 2, -1, -1], n = 3, and is_draft is True, then the
        function raises an exception, since there are not enough empty positions in the
        sequence for generating 3 tokens. If is_draft is False, the mask is
        [False, True, True, True, False].
        If tok_ids = [5, 3, 2, 1, 0], n=2, and is_draft is True, then the function
        raises an exception (and for any n > 0), since there are no empty positions in
        the sequence. If is_draft is False, the mask is
        [False, False, False, True, True].
        """
        empty_positions = torch.nonzero(self.tok_ids[0] == -1)
        if is_draft:
            start_idx = empty_positions[0]
            if start_idx + self.n > seq_len:
                raise Exception(
                    "Not enough tokens in sequence to generate response in draft mode."
                )
            mask = torch.zeros(seq_len, dtype=bool)
            mask[start_idx : start_idx + self.n] = True
            return mask
        end_idx = seq_len if not empty_positions.any() else empty_positions[0]
        if end_idx - self.n < 0:
            raise Exception("Not enough tokens in sequence to generate response.")
        mask = torch.zeros(seq_len, dtype=bool)
        mask[end_idx + 1 - self.n : end_idx + 1] = True  # Ensure non-negative index
        return mask


@dataclass
class Response(Message):
    request_timestamp: float
    is_draft: bool
    scores: torch.Tensor
    tok_ids: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"Response(id={self.id}, timestamp={self.timestamp}, "
            f"request_timestamp={self.request_timestamp}, "
            f"is_draft={self.is_draft}, scores_shape={self.scores.shape}, "
            f"tok_ids:\n{self.tok_ids})"
        )


@dataclass
class Preemption(Event):
    pass
