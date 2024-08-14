import logging
import time
from multiprocessing import Pipe, Queue
from threading import Thread

from transformers import AutoTokenizer

from dsi.online.actual.broker import broker, res_listener
from dsi.online.actual.model import Model
from dsi.online.actual.server import ServerDrafter, ServerTarget
from dsi.online.actual.state import State

log = logging.getLogger(__name__)


def main():
    res_receiver, res_sender = Pipe(duplex=False)
    verification_queue = Queue(maxsize=2)
    msg_bus = Queue()

    name_drafter = "facebook/opt-125m"
    name_target = "facebook/opt-350m"
    tokenizer_drafter = AutoTokenizer.from_pretrained(name_drafter)
    tokenizer_target = AutoTokenizer.from_pretrained(name_target)
    prompt = "Hello, my name is"
    state_drafter = State(tokenizer_drafter(prompt).input_ids)
    state_target = State(tokenizer_target(prompt).input_ids)
    print("States created.")
    model_drafter = Model(0, name_drafter, is_verifier=False, state=state_drafter)
    model_target1 = Model(1, name_target, is_verifier=True, state=state_target)
    model_target2 = Model(
        2, name_target, is_verifier=True, state=state_target.clone(only_verified=True)
    )
    print("Models created.")
    drafter = ServerDrafter(model_drafter, verification_queue, msg_bus, res_sender)
    target1 = ServerTarget(model_target1, verification_queue, msg_bus, None)
    target2 = ServerTarget(model_target2, verification_queue, msg_bus, None)
    servers = [drafter, target1, target2]
    print("Servers created.")
    # To allow servers to communicate with each other
    for server in servers:
        server.servers = servers
    # Start server processes
    th_servers = [Thread(target=server.run) for server in servers]
    print("Starting the broker and servers...")
    Thread(target=broker, args=(msg_bus, servers)).start()
    th_res = Thread(target=res_listener, args=(res_receiver,))
    th_res.start()
    start: float = time.time()
    for th in th_servers:
        th.start()
    res_sender.send(start)
    print("Waiting for the servers to finish...")
    th_res.join()
    for th in th_servers:
        th.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
