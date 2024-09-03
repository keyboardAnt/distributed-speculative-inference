# from torch.multiprocessing import Queue
import asyncio
import logging
import os

# import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    _job_queue: asyncio.Queue
    _msg_bus: asyncio.Queue
    # _result_pipe: asyncio.Future


class Server(ABC):
    _vocab_size: int = 100
    _S: int = 20

    def __init__(
        self,
        setup: SetupServer,
    ):
        self.setup: SetupServer = setup
        self.servers: list[Server] = []
        # self._preempted = threading.Event()
        self._result_pipe: asyncio.Future = asyncio.Future()

    # def _preempt(self) -> None:
    #     self._log("preempting current computation")
    #     self._preempted.set()

    # @final
    # def _is_preempted(self) -> bool:
    #     return self._preempted.is_set()

    # @final
    # def _wait_until_preemption(self) -> None:
    #     self._log("waiting until the server is preempted")
    #     self._preempted.wait()

    # @final
    # def _clear_preemption(self) -> None:
    #     self._log("clearing the preempted flag")
    #     self._preempted.clear()

    # @final
    # def _halt(self) -> None:
    #     self._log("Halting")
    #     ts: float = time.time()
    #     self.setup._result_pipe.send(ts)
    #     raise GenerationComplete(self.setup.model.setup.state.v)

    # @final
    # def _is_halted(self) -> bool:
    #     return self._halted.is_set()

    # @final
    # def _is_preempted(self) -> bool:
    #     return self._is_preempted() or self._is_halted()

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    async def cb_update_state(self, sender_id: int, m: MsgVerifiedRightmost) -> None:
        """
        Validates that the existing state aligns with the given last verified token.
        If the state is invalid, the server preempts and tries to roll back the state.
        If the rollback is successful, the server extends the state with the new token
        and updates the state's `v` index. If the rollback fails due to
        InvalidRollbackError, clones the state of the sender.
        """
        self._log("Callback to update state. Acquiring lock...")
        async with self.setup.model.setup.state.lock:
            self._log(f"Existing state: {self.setup.model.setup.state=}")
            self._log(f"Updating state given {m=}")
            if self.setup.model.setup.state.is_aligned(m):
                self._log("State is aligned")
                self.setup.model.setup.state.v = max(
                    m.v, self.setup.model.setup.state.v
                )
            else:
                self._log("State is not aligned")
                # self._preempt()
                try:
                    self._log(f"Rolling back to {m.v - 1}")
                    await self.setup.model.setup.state.rollback(m.v - 1)
                    self._log("Extending state...")
                    await self.setup.model.setup.state.extend([m.tok_id], verified=True)
                except InvalidRollbackError as e:
                    self._log(f"Invalid rollback: {e}")
                    # self._log(f"Cloning the state of {sender_id=}")
                    # self.setup.model.setup.state = self.servers[
                    #     sender_id
                    # ].setup.model.setup.state.clone(only_verified=True)
                    await self._clone_state_from(sender_id)
            self._log(f"State updated: {self.setup.model.setup.state=}")

    async def _clone_state_from(self, server_id: int) -> None:
        """Clones the state from another server."""
        self._log(f"Cloning the state of {server_id=}")
        self.setup.model.setup.state = await self.servers[
            server_id
        ].setup.model.setup.state.clone(only_verified=True)

    def _log(self, log_msg: str) -> None:
        pid = os.getpid()
        log.debug(
            f"[{pid=}] {self.__class__.__name__} -"
            f" GPU ID: {self.setup.model.setup.gpu_id} - {log_msg}"
        )
        # flush the log to ensure correct order
        logging.getLogger().handlers[0].flush()


class ServerDrafter(Server):
    def __init__(self, setup: SetupServer):
        super().__init__(setup)
        self._lookahead: int = 5

    # def _preempt(self) -> None:
    #     super()._preempt()
    #     self._log("Clearing the job queue")
    #     self.setup._job_queue.empty()

    # def _draft(self) -> list[int]:
    #     """Generates draft tokens. Returns their ids."""
    #     self._log("Acquiring lock to draft tokens...")
    #     with self.setup.model.setup.state.lock:
    #         curr_lookahead: int = min(
    #             self._lookahead,
    #             self._S - 1 - len(self.setup.model.setup.state.tok_ids)
    #         )
    #         self._log("Drafting tokens... ")
    #         tok_ids: list[int] = []
    #         if curr_lookahead > 0:
    #             tok_ids = self.setup.model.draft(curr_lookahead)
    #         self._log(f"Drafted: {tok_ids=}")
    #         self._log(f"State after drafting: {self.setup.model.setup.state=}")
    #         return tok_ids

    # def _halt(self) -> None:
    #     ts: float = time.time()
    #     self.setup._result_pipe.send(ts)
    #     raise GenerationComplete(self.setup.model.setup.state.v)

    # def run(self) -> None:
    #     """Returns the timestamp when the generation is complete."""
    #     while self.setup.model.setup.state.v < self._S:
    #         # TODO: Consider avoiding busy waiting when
    #         #       `len(self.setup.model.setup.state.tok_ids) == self._S`.
    #         #       Instead, wake up the drafter if the state is rolled back.
    #         tok_ids: list[int] = self._draft()
    #         if not self._is_preempted():
    #             self.setup._job_queue.put((self.setup.model.setup.gpu_id, tok_ids))
    #         else:
    #             self._resume()
    #     self.halt()

    async def run(self) -> None:
        """Returns the timestamp when the generation is complete."""
        while self.setup.model.setup.state.v < self._S:
            tok_ids: list[int] = []
            self._log("Acquiring lock to draft tokens...")
            async with self.setup.model.setup.state.lock:
                curr_lookahead: int = min(
                    self._lookahead,
                    self._S - 1 - len(self.setup.model.setup.state.tok_ids),
                )
                self._log("Drafting tokens... ")
                if curr_lookahead > 0:
                    tok_ids = await self.setup.model.draft(curr_lookahead)
                self._log(f"Drafted: {tok_ids=}")
                self._log(f"State after drafting: {self.setup.model.setup.state=}")
            # if not self._is_preempted():
            #     self.setup._job_queue.put((self.setup.model.setup.gpu_id, tok_ids))
            #     if not tok_ids:
            #         self._wait_on_preemption()
            # else:
            #     self._clear_preemption()
            # if self._is_preempted():
            #     self._log("preempted and will not send job to queue.")
            #     self._clear_preemption()
            # else:
            # await self.setup._job_queue.put((self.setup.model.setup.gpu_id, tok_ids))
            # if not tok_ids:
            #     self._wait_until_preemption()
            await self.setup._job_queue.put((self.setup.model.setup.gpu_id, tok_ids))


class ServerTarget(Server):
    async def run(self) -> None:
        sender_id: int
        tok_ids: list[int]
        while True:
            self._log("Reading from the job queue... If empty, will block.")
            sender_id, tok_ids = await self.setup._job_queue.get()
            self._log(f"Received tokens from {sender_id=}: {tok_ids=}")
            msg: None | MsgVerifiedRightmost = await self._verify(tok_ids)
            # if self._is_preempted():
            #     self._clear_preemption()
            # elif msg.v >= self._S:
            #     self._halt()
            # else:
            #     self.setup._msg_bus.put((self.setup.model.setup.gpu_id, msg))
            if msg and msg.v >= self._S:
                self.setup._result_pipe.set_result(time.time())
                raise GenerationComplete(self.setup.model.setup.state.v)

    async def _verify(self, tok_ids: list[int]) -> MsgVerifiedRightmost:
        """
        Verifies the given drafts.
        Returns the token id and index of the last verified token.
        """
        # if self._is_preempted():
        #     self._log("preempted.")
        #     return None
        self._log(f"Verifying: {tok_ids=}")
        msg: MsgVerifiedRightmost = await self.setup.model.verify(tok_ids)
        self._log(f"Verified: {msg=}")
        self._log(f"New state: {self.setup.model.setup.state=}")
        return msg
