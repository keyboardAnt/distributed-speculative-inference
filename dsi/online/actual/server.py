import random
from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import final

import torch

from dsi.online.actual.message import Message
from dsi.online.actual.state import State


class Server(ABC):
    @final
    def __init__(self, gpu_id: int, queue: Queue, message_bus: Queue):
        self._gpu_id: int = gpu_id
        self._queue: Queue = queue
        self._message_bus: Queue = message_bus
        self._state: State = State([])  # Empty initial state

    # TODO: Implement this method
    @final
    def _preempt(self) -> None:
        print(f"GPU {self._gpu_id} preempting current computation.")

    @final
    def run(self) -> None:
        torch.cuda.set_device(self._gpu_id)
        while True:
            self._run()

    def update_state(self, message: Message) -> None:
        self._preempt()
        self._state.update(message.i, message.tok_id)

    @abstractmethod
    def _run(self) -> None:
        raise NotImplementedError


class ServerDrafter(Server):
    def __init__(self, gpu_id, queue, message_bus):
        super().__init__(gpu_id, queue, message_bus)

    # TODO: Implement this method
    def _draft(self) -> list[int]:
        """Generates 10 draft tokens. Returns their ids."""
        return [random.randint(1, 100) for _ in range(10)]

    def _run(self) -> None:
        tok_ids: list[int] = self._draft()
        self._state.extend(tok_ids)
        self._queue.put((self._gpu_id, tok_ids))


class ServerTarget(Server):
    def __init__(self, gpu_id, queue, message_bus):
        super().__init__(gpu_id, queue, message_bus)

    # TODO: Implement this method
    def _verify(self, tok_ids) -> tuple[int, int]:
        """
        Verifies the given drafts.
        Returns the token id and index of the last verified token.
        """
        verified = random.choice([True, False])
        i: int = len(tok_ids) - 1 if verified else len(tok_ids) - 2
        self._state.update(i, tok_ids[i])
        if verified:
            return tok_ids[-1], len(tok_ids) - 1
        return tok_ids[-2], len(tok_ids) - 2

    def _run(self):
        sender_id, tokens = self._queue.get()
        result_tok_id, result_index = self._verify(tokens)
        message = Message(result_tok_id, result_index)
        self._message_bus.put((self._gpu_id, message))
