import logging
from queue import Queue
from threading import Thread

from dsi.online.actual.broker import broker
from dsi.online.actual.server import ServerDrafter, ServerTarget
from dsi.online.actual.state import State

log = logging.getLogger(__name__)


def main():
    # manager = Manager()
    verification_queue = Queue(maxsize=2)
    msg_bus = Queue()

    state = State([2, 0, 2, 4])
    drafter = ServerDrafter(0, state, verification_queue, msg_bus)
    target1 = ServerTarget(
        1, state.clone(only_verified=True), verification_queue, msg_bus
    )
    target2 = ServerTarget(
        2, state.clone(only_verified=True), verification_queue, msg_bus
    )

    servers = [drafter, target1, target2]
    # To allow servers to communicate with each other
    for server in servers:
        server.servers = servers
    # Start server processes
    threads = [Thread(target=server.run) for server in servers] + [
        Thread(target=broker, args=(msg_bus, servers))
    ]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
