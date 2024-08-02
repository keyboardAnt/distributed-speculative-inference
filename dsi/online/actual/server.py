import logging
import os
import random
import time
from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import final

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.state import State
from dsi.utils import set_random_seed

log = logging.getLogger(__name__)

set_random_seed(0)


class Server(ABC):
    @final
    def __init__(self, gpu_id: int, queue: Queue, msg_bus: Queue, state: State):
        self.gpu_id: int = gpu_id
        self._queue: Queue = queue
        self._msg_bus: Queue = msg_bus
        self._state: State = state  # Empty initial state

    # TODO: Implement this method
    @final
    def _preempt(self) -> None:
        print(f"GPU {self.gpu_id} preempting current computation.")

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
        self._log(f"Existing state: {self._state=}")
        self._log(f"Updating state with {m=}")
        if self._state.is_aligned(m):
            self._log("State is aligned")
            self._state.v = m.v
            self._log(f"State updated: {self._state=}")
        else:
            self._log("State is not aligned")
            self._preempt()
            self._log(f"Rolling back to {m.v - 1}")
            self._state.rollback(m.v - 1)
            self._log("Extending state...")
            self._state.extend([m.tok_id], verified=True)
            self._log(f"State updated: {self._state=}")

    @abstractmethod
    def _run(self) -> None:
        raise NotImplementedError

    @staticmethod
    def msg_listener(msg_bus: Queue, servers: list["Server"]):
        while True:
            msg: MsgVerifiedRightmost
            sender_id, msg = msg_bus.get()
            state = State.from_dict(msg.state)  # Deserialize state
            log.debug(f"[LISTENER] New message from {sender_id=}: {msg=}, {state=}")
            for server in servers:
                if (
                    server.gpu_id != sender_id
                ):  # Assuming servers only react to messages from others
                    server.cb_update_state(msg)

    def _log(self, log_msg: str) -> None:
        pid = os.getpid()
        log.debug(
            f"[{pid=}] {self.__class__.__name__} - GPU ID: {self.gpu_id} - {log_msg}"
        )
        # flush the log to ensure that the message is printed in the correct order
        logging.getLogger().handlers[0].flush()


class ServerDrafter(Server):
    _lookahead: int = 5

    # TODO: Implement this method
    def _draft(self) -> list[int]:
        """Generates 10 draft tokens. Returns their ids."""
        self._log("Drafting tokens...")
        time.sleep(0.1)
        tok_ids: list[int] = [random.randint(1, 100) for _ in range(self._lookahead)]
        self._log(f"Drafted: {tok_ids=}")
        self._log("Extending state...")
        self._state.extend(tok_ids, verified=False)
        self._log(f"State updated: {self._state=}")
        return tok_ids

    def _run(self) -> None:
        while self._state.v < 100 and not self._queue.full():
            tok_ids: list[int] = self._draft()
            self._queue.put((self.gpu_id, tok_ids))


class ServerTarget(Server):
    # TODO: Implement this method
    def _verify(self, tok_ids: list[int]) -> MsgVerifiedRightmost:
        """
        Verifies the given drafts.
        Returns the token id and index of the last verified token.
        """
        tok_id_extra: int = random.randint(1, 100)
        self._log(f"Verifying tokens {tok_ids=} + {tok_id_extra=}")
        tok_ids.append(tok_id_extra)
        time.sleep(1)
        num_verified: int = random.randint(1, len(tok_ids))
        tok_id_verified_rightmost: int = tok_ids[num_verified - 1]
        self._state.extend(tok_ids[:num_verified], verified=True)
        self._log(f"{num_verified=}")
        self._log(f"Extended the state. New state: {self._state=}")
        return MsgVerifiedRightmost(
            v=self._state.v,
            tok_id=tok_id_verified_rightmost,
        )

    def _run(self):
        sender_id: int
        tok_ids: list[int]
        while True:
            sender_id, tok_ids = self._queue.get()
            msg: MsgVerifiedRightmost = self._verify(tok_ids)
            self._msg_bus.put((self.gpu_id, msg))
