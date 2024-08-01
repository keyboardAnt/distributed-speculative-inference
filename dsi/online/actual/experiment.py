import copy
import logging
from multiprocessing import Process, Queue

from dsi.online.actual.server import Server, ServerDrafter, ServerTarget
from dsi.online.actual.state import State


def main():
    verification_queue = Queue(maxsize=2)
    msg_bus = Queue()

    prompt: list[int] = [2, 0, 2, 4]
    state = State(copy.deepcopy(prompt))
    drafter = ServerDrafter(0, verification_queue, msg_bus, state)
    target1 = ServerTarget(1, verification_queue, msg_bus, copy.deepcopy(state))
    target2 = ServerTarget(2, verification_queue, msg_bus, copy.deepcopy(state))

    servers = [drafter, target1, target2]

    # Start server processes
    processes = [Process(target=server.run) for server in servers] + [
        Process(target=Server.msg_listener, args=(msg_bus, servers))
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
