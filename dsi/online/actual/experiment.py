import logging
from multiprocessing import Process, Queue

from dsi.online.actual.server import Server, ServerDrafter, ServerTarget


def main():
    verification_queue = Queue()
    msg_bus = Queue()

    drafter = ServerDrafter(0, verification_queue, msg_bus)
    target1 = ServerTarget(1, verification_queue, msg_bus)
    target2 = ServerTarget(2, verification_queue, msg_bus)

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
