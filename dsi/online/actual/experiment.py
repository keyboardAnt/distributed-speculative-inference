# from torch.multiprocessing import Pipe, Process, Queue
import asyncio
import logging

from transformers import AutoTokenizer

from dsi.online.actual.broker import Broker
from dsi.online.actual.model import Model, SetupModel
from dsi.online.actual.server import ServerDrafter, ServerTarget, SetupServer
from dsi.online.actual.state import State

# from threading import Thread


log = logging.getLogger(__name__)


def tokenize(tokenizer: str, prompt: str) -> list[int]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer(prompt).input_ids


# def main():
#     res_receiver, res_sender = Pipe(duplex=False)
#     job_queue = Queue(maxsize=2)
#     msg_bus = Queue()

#     name_drafter = "facebook/opt-125m"
#     name_target = "facebook/opt-350m"
#     prompt = "Hello, my name is"
#     prompt_drafter: list[int] = tokenize(name_drafter, prompt)
#     prompt_target: list[int] = tokenize(name_target, prompt)
#     state_drafter = State(prompt_drafter)
#     state_target = State(prompt_target)
#     setup_model_drafter = SetupModel(gpu_id=0, _name=name_drafter,state=state_drafter)
#     setup_model_target1 = SetupModel(gpu_id=1, _name=name_target, state=state_target)
#     setup_model_target2 = SetupModel(
#         gpu_id=2, _name=name_target, state=state_target.clone(only_verified=True)
#     )
#     model_drafter = Model(setup_model_drafter)
#     model_target1 = Model(setup_model_target1)
#     model_target2 = Model(setup_model_target2)
#     print("Models created.")
#     setup_server_drafter = SetupServer(
#         model=model_drafter,
#         _job_queue=job_queue,
#         _msg_bus=msg_bus,
#         _result_pipe=res_sender,
#     )
#     setup_server_target1 = SetupServer(
#         model=model_target1,
#         _job_queue=job_queue,
#         _msg_bus=msg_bus,
#         _result_pipe=res_sender,
#     )
#     setup_server_target2 = SetupServer(
#         model=model_target2,
#         _job_queue=job_queue,
#         _msg_bus=msg_bus,
#         _result_pipe=res_sender,
#     )
#     drafter = ServerDrafter(setup_server_drafter)
#     target1 = ServerTarget(setup_server_target1)
#     target2 = ServerTarget(setup_server_target2)
#     servers = [drafter, target1, target2]
#     print("Servers created.")
#     # To allow servers to communicate with each other
#     for server in servers:
#         server.servers = servers
#     # Start servers
#     # TODO(#44): Multiprocess support
#     th_servers = [Thread(target=server.run) for server in servers]
#     print("Starting the broker and servers...")
#     broker = Broker(msg_bus, servers)
#     # Thread(target=broker.run).start()
#     # th_res = Thread(target=res_listener, args=(res_receiver,))
#     Process(target=broker.run).start()
#     th_res = Process(target=res_listener, args=(res_receiver,))
#     th_res.start()
#     start: float = time.time()
#     for th in th_servers:
#         th.start()
#     res_sender.send(start)
#     print("Waiting for the servers to finish...")
#     th_res.join()
#     for th in th_servers:
#         th.join()


async def main():
    # res_receiver, res_sender = Pipe(duplex=False)
    job_queue = asyncio.Queue(maxsize=2)
    msg_bus = asyncio.Queue()

    name_drafter = "facebook/opt-125m"
    name_target = "facebook/opt-350m"

    prompt = "Hello, my name is"
    prompt_drafter: list[int] = tokenize(name_drafter, prompt)
    prompt_target: list[int] = tokenize(name_target, prompt)

    state_drafter = State(prompt_drafter)
    state_target = State(prompt_target)

    setup_model_drafter = SetupModel(gpu_id=0, _name=name_drafter, state=state_drafter)
    setup_model_target1 = SetupModel(gpu_id=1, _name=name_target, state=state_target)
    setup_model_target2 = SetupModel(
        gpu_id=2, _name=name_target, state=state_target.clone(only_verified=True)
    )

    model_drafter = Model(setup_model_drafter)
    model_target1 = Model(setup_model_target1)
    model_target2 = Model(setup_model_target2)
    print("Models created.")

    setup_server_drafter = SetupServer(
        model=model_drafter,
        _job_queue=job_queue,
        _msg_bus=msg_bus,
        # _result_pipe=res_sender,
    )
    setup_server_target1 = SetupServer(
        model=model_target1,
        _job_queue=job_queue,
        _msg_bus=msg_bus,
        # _result_pipe=res_sender,
    )
    setup_server_target2 = SetupServer(
        model=model_target2,
        _job_queue=job_queue,
        _msg_bus=msg_bus,
        # _result_pipe=res_sender,
    )

    drafter = ServerDrafter(setup_server_drafter)
    target1 = ServerTarget(setup_server_target1)
    target2 = ServerTarget(setup_server_target2)
    servers = [drafter, target1, target2]
    print("Servers created.")
    # To allow servers to communicate with each other
    for server in servers:
        server.servers = servers

    print("Starting the broker and servers...")
    broker = Broker(msg_bus, servers)
    tasks = [asyncio.create_task(server.run()) for server in servers] + [
        asyncio.create_task(broker.run())
    ]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # main()
    asyncio.run(main())
