import logging
import time
from multiprocessing import Pipe
from queue import Queue
from threading import Thread

from dsi.online.actual.broker import broker, res_listener
from dsi.online.actual.server import ServerDrafter, ServerTarget
from dsi.online.actual.state import State

log = logging.getLogger(__name__)


def main():
    res_receiver, res_sender = Pipe(duplex=False)
    verification_queue = Queue(maxsize=2)
    msg_bus = Queue()

    state = State([2, 0, 2, 4])
    drafter = ServerDrafter(0, state, verification_queue, msg_bus, res_sender)
    target1 = ServerTarget(
        1, state.clone(only_verified=True), verification_queue, msg_bus, None
    )
    target2 = ServerTarget(
        2, state.clone(only_verified=True), verification_queue, msg_bus, None
    )
    servers = [drafter, target1, target2]
    # To allow servers to communicate with each other
    for server in servers:
        server.servers = servers
    # Start server processes
    th_servers = [Thread(target=server.run) for server in servers]
    th_res = Thread(target=res_listener, args=(res_receiver,))
    th_res.start()
    Thread(target=broker, args=(msg_bus, servers)).start()
    start: float = time.time()
    for th in th_servers:
        th.start()
    res_sender.send(start)
    th_res.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
