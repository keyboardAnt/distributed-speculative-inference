import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import final

from torch.multiprocessing import Queue

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.model import Model
from dsi.online.actual.state import InvalidRollbackError
from dsi.utils import set_random_seed

log = logging.getLogger(__name__)

set_random_seed(0)


class GenerationComplete(Exception):
    def __init__(self, S: int):
        super().__init__(f"Completed generating {S} tokens.")


@dataclass
class SetupServer:
    model: Model
    _job_queue: Queue
    _msg_bus: Queue
    _result_pipe: any


class Server(ABC):
    _vocab_size: int = 100
    _S: int = 20

    @final
    def __init__(
        self,
        setup: SetupServer,
    ):
        self.setup: SetupServer = setup
        self.servers: list[Server] = []
        self._preempted = threading.Event()
        self._halted = threading.Event()

    def preempt(self) -> None:
        self._log("preempting current computation")
        self._preempted.set()

    @final
    def _is_preempted(self) -> bool:
        return self._preempted.is_set()

    @final
    def _resume(self) -> None:
        self._log("clearing the preempted flag if not halted")
        if self._is_halted():
            raise GenerationComplete(self.setup.model.setup.state.v)
        self._preempted.clear()

    def halt(self) -> None:
        self._log("Halting")
        self._halted.set()

    @final
    def _is_halted(self) -> bool:
        return self._halted.is_set()

    @final
    def _is_preempted_or_halted(self) -> bool:
        return self._is_preempted() or self._is_halted()

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    def cb_update_state(self, sender_id: int, m: MsgVerifiedRightmost) -> None:
        """
        Validates that the existing state aligns with the given last verified token.
        If the state is invalid, the server preempts and tries to roll back the state.
        If the rollback is successful, the server extends the state with the new token
        and updates the state's `v` index. If the rollback fails due to
        InvalidRollbackError, clones the state of the sender.
        """
        self._log("Callback to update state. Acquiring lock...")
        with self.setup.model.setup.state.lock:
            self._log(f"Existing state: {self.setup.model.setup.state=}")
            self._log(f"Updating state with {m=}")
            if self.setup.model.setup.state.is_aligned(m):
                self._log("State is aligned")
                self.setup.model.setup.state.v = m.v
            else:
                self._log("State is not aligned")
                self.preempt()
                try:
                    self._log(f"Rolling back to {m.v - 1}")
                    self.setup.model.setup.state.rollback(m.v - 1)
                    self._log("Extending state...")
                    self.setup.model.setup.state.extend([m.tok_id], verified=True)
                except InvalidRollbackError as e:
                    self._log(f"Invalid rollback: {e}")
                    self._log(f"Cloning the state of {sender_id=}")
                    self.setup.model.setup.state = self.servers[
                        sender_id
                    ].setup.model.setup.state.clone(only_verified=True)
            self._log(f"State updated: {self.setup.model.setup.state=}")

    def _log(self, log_msg: str) -> None:
        pid = os.getpid()
        log.debug(
            f"[{pid=}] {self.__class__.__name__} -"
            f" GPU ID: {self.setup.model.setup.gpu_id} - {log_msg}"
        )
        # flush the log to ensure correct order
        logging.getLogger().handlers[0].flush()


class ServerDrafter(Server):
    _lookahead: int = 5

    def preempt(self) -> None:
        super().preempt()
        self._log("Clearing the job queue")
        self.setup._job_queue.empty()

    def _draft(self) -> list[int]:
        """Generates draft tokens. Returns their ids."""
        self._log("Acquiring lock to draft tokens...")
        with self.setup.model.setup.state.lock:
            curr_lookahead: int = min(
                self._lookahead, self._S - 1 - len(self.setup.model.setup.state.tok_ids)
            )
            self._log("Drafting tokens... ")
            tok_ids: list[int] = []
            if curr_lookahead > 0:
                tok_ids = self.setup.model.draft(curr_lookahead)
            self._log(f"Drafted: {tok_ids=}")
            self._log(f"State after drafting: {self.setup.model.setup.state=}")
            return tok_ids

    def halt(self) -> None:
        ts: float = time.time()
        self.setup._result_pipe.send(ts)
        super().halt()
        self._log("Halting other servers")
        for server in self.servers[1:]:
            server.halt()
        self._log("Clearing the job queue and message bus")
        self.setup._job_queue.empty()
        self.setup._msg_bus.empty()
        raise GenerationComplete(self.setup.model.setup.state.v)

    def run(self) -> None:
        """Returns the timestamp when the generation is complete."""
        while self.setup.model.setup.state.v < self._S:
            # TODO: Consider avoiding busy waiting when
            #       `len(self.setup.model.setup.state.tok_ids) == self._S`.
            #       Instead, wake up the drafter if the state is rolled back.
            tok_ids: list[int] = self._draft()
            if not self._is_preempted():
                self.setup._job_queue.put((self.setup.model.setup.gpu_id, tok_ids))
            else:
                self._resume()
        self.halt()


class ServerTarget(Server):
    def _verify(self, tok_ids: list[int]) -> MsgVerifiedRightmost:
        """
        Verifies the given drafts.
        Returns the token id and index of the last verified token.
        """
        if self._is_preempted_or_halted():
            self._log("preempted or halted.")
            return None
        self._log(f"Verifying: {tok_ids=}")
        msg: MsgVerifiedRightmost = self.setup.model.verify(tok_ids)
        self._log(f"Verified: {msg=}")
        self._log(f"New state: {self.setup.model.setup.state=}")
        return msg

    def run(self) -> None:
        sender_id: int
        tok_ids: list[int]
        while not self._is_halted():
            self._log("Reading from the job queue... If empty, will block.")
            sender_id, tok_ids = self.setup._job_queue.get()
            self._log(f"Received tokens from {sender_id=}: {tok_ids=}")
            msg: None | MsgVerifiedRightmost = self._verify(tok_ids)
            if self._is_preempted():
                self._resume()
            else:
                self.setup._msg_bus.put((self.setup.model.setup.gpu_id, msg))
