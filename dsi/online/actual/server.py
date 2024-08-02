import logging
import os
import random
import threading
import time
from abc import ABC, abstractmethod
from queue import Queue
from typing import final

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.state import InvalidRollbackError, State
from dsi.utils import set_random_seed

log = logging.getLogger(__name__)

set_random_seed(0)


class Server(ABC):
    @final
    def __init__(
        self,
        gpu_id: int,
        state: State,
        queue: Queue,
        msg_bus: Queue,
    ):
        self.gpu_id: int = gpu_id
        self.state: State = state
        self._queue: Queue = queue
        self._msg_bus: Queue = msg_bus
        self.servers: list[Server] = []
        self._preempted = threading.Event()
        self._lock = threading.Lock()

    @final
    def _preempt(self) -> None:
        self._log("preempting current computation")
        self._preempted.set()

    @final
    def _is_preempted(self) -> bool:
        return self._preempted.is_set()

    @final
    def _resume(self) -> None:
        self._log("clearing the preempted flag")
        self._preempted.clear()

    @final
    def run(self) -> None:
        # TODO: use the GPU
        # torch.cuda.set_device(self._gpu_id)
        self._run()

    @abstractmethod
    def _run(self) -> None:
        raise NotImplementedError

    def cb_update_state(self, sender_id: int, m: MsgVerifiedRightmost) -> None:
        """
        Validates that the existing state aligns with the given last verified token.
        If the state is invalid, the server preempts and tries to roll back the state.
        If the rollback is successful, the server extends the state with the new token
        and updates the state's `v` index. If the rollback fails due to
        InvalidRollbackError, clones the state of the sender.
        """
        self._log(f"Existing state: {self.state=}")
        self._log(f"Updating state with {m=}")
        if self.state.is_aligned(m):
            self._log("State is aligned")
            self.state.v = m.v
        else:
            self._log("State is not aligned")
            self._preempt()
            try:
                self._log(f"Rolling back to {m.v - 1}")
                self.state.rollback(m.v - 1)
                self._log("Extending state...")
                self.state.extend([m.tok_id], verified=True)
            except InvalidRollbackError as e:
                self._log(f"Invalid rollback: {e}")
                self._log(f"Cloning the state of {sender_id=}")
                self.state = self.servers[sender_id].state.clone(only_verified=True)
        self._log(f"State updated: {self.state=}")

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
        if self._is_preempted():
            self._log("interrupted.")
            return []
        tok_ids: list[int] = [random.randint(1, 100) for _ in range(self._lookahead)]
        self._log(f"Drafted: {tok_ids=}")
        self._log("Extending state...")
        self.state.extend(tok_ids, verified=False)
        self._log(f"State updated: {self.state=}")
        return tok_ids

    def _run(self) -> None:
        while self.state.v < 100 and not self._queue.full():
            tok_ids: list[int] = self._draft()
            if self._is_preempted():
                self._resume()
            else:
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
        if self._is_preempted():
            self._log("interrupted.")
            return None
        num_verified: int = random.randint(1, len(tok_ids))
        tok_id_verified_rightmost: int = tok_ids[num_verified - 1]
        self.state.extend(tok_ids[:num_verified], verified=True)
        self._log(f"{num_verified=}")
        self._log(f"Extended the state. New state: {self.state=}")
        return MsgVerifiedRightmost(
            v=self.state.v,
            tok_id=tok_id_verified_rightmost,
        )

    def _run(self):
        sender_id: int
        tok_ids: list[int]
        while True:
            sender_id, tok_ids = self._queue.get()
            msg: None | MsgVerifiedRightmost = self._verify(tok_ids)
            if self._is_preempted():
                self._resume()
            else:
                self._msg_bus.put((self.gpu_id, msg))
