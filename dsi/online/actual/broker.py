import logging
from queue import Queue

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.server import Server

log = logging.getLogger(__name__)


def broker(bus: Queue, servers: list[Server]) -> None:
    """
    Listens the message bus. Broadcasts messages to all servers except the sender.
    """
    sender_id: int
    msg: MsgVerifiedRightmost
    while True:
        sender_id, msg = bus.get()
        log.debug(
            "[LISTENER] New message from sender_id=%d: msg=%s, state=%s",
            sender_id,
            msg,
            servers[sender_id].model.state,
        )
        for server in servers:
            if (
                server.model.gpu_id != sender_id
            ):  # Assuming servers only react to messages from others
                server.cb_update_state(sender_id, msg)


def res_listener(pipe) -> float:
    start: float = pipe.recv()
    log.info(f"Timestamp start: {start=}")
    end: float = pipe.recv()
    log.info(f"Timestamp end: {start=}")
    latency: float = end - start
    log.info(f"Latency: {latency:.3f} seconds")
    return latency
