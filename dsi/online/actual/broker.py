# from torch.multiprocessing import Queue
import asyncio
import logging

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.server import Server

log = logging.getLogger(__name__)


class Broker:
    def __init__(self, bus: asyncio.Queue, servers: list[Server]) -> None:
        self._bus: asyncio.Queue = bus
        self._servers: list[Server] = servers

    async def run(self) -> None:
        """
        Listens to the message bus.
        Broadcasts messages to all servers except the sender.
        """
        sender_id: int
        msg: MsgVerifiedRightmost
        while True:
            sender_id, msg = await self._most_recent()
            log.debug(
                "[LISTENER] New message from sender_id=%d: msg=%s, state=%s",
                sender_id,
                msg,
                self._servers[sender_id].setup.model.setup.state,
            )
            for server in self._servers:
                if (
                    server.setup.model.setup.gpu_id != sender_id
                ):  # Assuming servers only react to messages from others
                    await server.cb_update_state(sender_id, msg)

    async def _most_recent(self) -> tuple[int, MsgVerifiedRightmost]:
        """
        Retrieves the most recent sender id and message from the message bus.
        """
        sender_id: int
        msg: MsgVerifiedRightmost
        sender_id, msg = await self._bus.get()
        while not self._bus.empty():
            log.debug(
                "[LISTENER] Checking for more recent messages."
                f" Current message: {msg=}"
            )
            sender_id_temp: int
            msg_temp: MsgVerifiedRightmost
            sender_id_temp, msg_temp = await self._bus.get()
            if msg_temp.v > msg.v:
                msg = msg_temp
                sender_id = sender_id_temp
                log.debug(
                    f"[LISTENER] Discarding outdated message "
                    f"({msg_temp.v=} > {msg.v=}). Messages:\n{msg_temp=}\n{msg=}"
                )
            log.debug(f"[LISTENER] Returning most recent message: {msg=}")
        return sender_id, msg


# def broker(bus: Queue, servers: list[Server]) -> None:
#     """
#     Listens the message bus. Broadcasts messages to all servers except the sender.
#     """
#     v: int = -1
#     sender_id: int
#     msg: MsgVerifiedRightmost
#     while True:
#         sender_id, msg = bus.get()
#         log.debug(
#             "[LISTENER] New message from sender_id=%d: msg=%s, state=%s",
#             sender_id,
#             msg,
#             servers[sender_id].setup.model.setup.state,
#         )
#         if msg.v > v:
#             v = msg.v
#             for server in servers:
#                 if (
#                     server.setup.model.setup.gpu_id != sender_id
#                 ):  # Assuming servers only react to messages from others
#                     server.cb_update_state(sender_id, msg)
#         else:
#             log.debug(f"[LISTENER] Ignoring outdated message. {msg.v=}, {v=}")


def res_listener(pipe) -> float:
    start: float = pipe.recv()
    log.info(f"Timestamp start: {start=}")
    end: float = pipe.recv()
    log.info(f"Timestamp end: {start=}")
    latency: float = end - start
    log.info(f"Latency: {latency:.3f} seconds")
    return latency
