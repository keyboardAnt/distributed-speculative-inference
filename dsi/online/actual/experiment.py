import logging
import time
from multiprocessing import Pipe, Process, Queue
from threading import Thread

from transformers import AutoTokenizer

from dsi.online.actual.broker import broker, res_listener
from dsi.online.actual.model import Model, SetupModel
from dsi.online.actual.server import ServerDrafter, ServerTarget, SetupServer
from dsi.online.actual.state import State

# from threading import Thread


log = logging.getLogger(__name__)


def tokenize(tokenizer: str, prompt: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer(prompt).input_ids


def main():
    res_receiver, res_sender = Pipe(duplex=False)
    job_queue = Queue(maxsize=2)
    msg_bus = Queue()

    name_drafter = "facebook/opt-125m"
    name_target = "facebook/opt-350m"
    prompt = "Hello, my name is"
    prompt_drafter: list[int] = tokenize(name_drafter, prompt)
    prompt_target: list[int] = tokenize(name_target, prompt)
    state_drafter = State(prompt_drafter)
    state_target = State(prompt_target)
    setup_model_drafter = SetupModel(gpu_id=0, _name=name_drafter, state=state_drafter)
    setup_model_target1 = SetupModel(gpu_id=1, _name=name_target, state=state_target)
    setup_model_target2 = SetupModel(gpu_id=2, _name=name_target, state=state_target)
    model_drafter = Model(setup_model_drafter)
    model_target1 = Model(setup_model_target1)
    model_target2 = Model(setup_model_target2)
    print("Models created.")
    setup_server_drafter = SetupServer(
        model=model_drafter,
        _job_queue=job_queue,
        _msg_bus=msg_bus,
        _result_pipe=res_sender,
    )
    setup_server_target1 = SetupServer(
        model=model_target1, _job_queue=job_queue, _msg_bus=msg_bus, _result_pipe=None
    )
    setup_server_target2 = SetupServer(
        model=model_target2, _job_queue=job_queue, _msg_bus=msg_bus, _result_pipe=None
    )
    drafter = ServerDrafter(setup_server_drafter)
    target1 = ServerTarget(setup_server_target1)
    target2 = ServerTarget(setup_server_target2)
    servers = [drafter, target1, target2]
    print("Servers created.")
    # To allow servers to communicate with each other
    for server in servers:
        server.servers = servers
    # Start server processes
    pr_servers = [Process(target=server.run) for server in servers]
    print("Starting the broker and servers...")
    Thread(target=broker, args=(msg_bus, servers)).start()
    th_res = Thread(target=res_listener, args=(res_receiver,))
    th_res.start()
    start: float = time.time()
    for pr in pr_servers:
        pr.start()
    res_sender.send(start)
    print("Waiting for the servers to finish...")
    th_res.join()
    for pr in pr_servers:
        pr.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
