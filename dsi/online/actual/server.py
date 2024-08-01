import logging
import random
from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import final

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.state import State

log = logging.getLogger(__name__)


class Server(ABC):
    @final
    def __init__(self, gpu_id: int, queue: Queue, msg_bus: Queue):
        self._gpu_id: int = gpu_id
        self._queue: Queue = queue
        self._msg_bus: Queue = msg_bus
        self._state: State = State([])  # Empty initial state

    # TODO: Implement this method
    @final
    def _preempt(self) -> None:
        print(f"GPU {self._gpu_id} preempting current computation.")

    @final
    def run(self) -> None:
        # TODO: use the GPU
        # torch.cuda.set_device(self._gpu_id)
        self._run()

    def cb_update_state(self, m: MsgVerifiedRightmost) -> None:
        """
        Validates that the existing state aligns with the given last verified token.
        If the state is invalid, the server preempts and rolls back the state.
        Updates the state's `v` index.
        """
        log.debug("Server %d received message %s", self._gpu_id, m)
        if not self._state.is_aligned(m):
            self._preempt()
            self._state.rollback(m.v - 1)
        self._state.extend([m.tok_id], verified=True)

    @abstractmethod
    def _run(self) -> None:
        raise NotImplementedError

    @staticmethod
    def msg_listener(msg_bus: Queue, servers: list["Server"]):
        while True:
            sender_id: int
            msg: MsgVerifiedRightmost
            sender_id, msg = msg_bus.get()
            for server in servers:
                server.cb_update_state(msg)


class ServerDrafter(Server):
    # TODO: Implement this method
    def _draft(self) -> list[int]:
        """Generates 10 draft tokens. Returns their ids."""
        log.debug("Drafting tokens")
        return [random.randint(1, 100) for _ in range(10)]

    def _run(self) -> None:
        while self._state.v < 100:
            log.debug(f"Server {self._gpu_id} with {self._state.v=}")
            tok_ids: list[int] = self._draft()
            self._queue.put((self._gpu_id, tok_ids))
            self._state.extend(tok_ids, verified=False)


class ServerTarget(Server):
    # TODO: Implement this method
    def _verify(self, tok_ids) -> MsgVerifiedRightmost:
        """
        Verifies the given drafts.
        Returns the token id and index of the last verified token.
        """
        log.debug("Verifying tokens %s", tok_ids)
        verified = random.choice([True, False])
        i: int = len(tok_ids) - 1 if verified else len(tok_ids) - 2
        self._state.v += i
        return MsgVerifiedRightmost(self._state.v, tok_ids[i])

    def _run(self):
        sender_id: int
        tok_ids: list[int]
        while True:
            sender_id, tok_ids = self._queue.get()
            msg: MsgVerifiedRightmost = self._verify(tok_ids)
            self._msg_bus.put((self._gpu_id, msg))
