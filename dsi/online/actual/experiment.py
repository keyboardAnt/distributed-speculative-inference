from multiprocessing import Process, Queue

from dsi.online.actual.message import message_listener
from dsi.online.actual.server import ServerDrafter, ServerTarget


def main():
    verification_queue = Queue()
    message_bus = Queue()

    drafter = ServerDrafter(0, verification_queue, message_bus)
    target1 = ServerTarget(1, verification_queue, message_bus)
    target2 = ServerTarget(2, verification_queue, message_bus)

    servers = [drafter, target1, target2]

    # Start server processes
    processes = [Process(target=server.run) for server in servers] + [
        Process(target=message_listener, args=(message_bus, servers))
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()


if __name__ == "__main__":
    main()
