import asyncio
import time
from dataclasses import dataclass
from typing import Set
from uuid import UUID, uuid4

import aioconsole
import torch
from transformers import AutoModelForCausalLM


@dataclass
class Event:
    timestamp: float

    @classmethod
    def create(cls) -> "Event":
        return cls(timestamp=time.time())


@dataclass
class Message(Event):
    id: UUID

    @classmethod
    def create(cls) -> "Message":
        return cls(id=uuid4(), timestamp=time.time())


@dataclass
class Request(Message):
    tok_ids: torch.Tensor
    n: int

    @classmethod
    def create(cls, tok_ids: torch.Tensor, n: int) -> "Request":
        return cls(id=uuid4(), timestamp=time.time(), tok_ids=tok_ids, n=n)


@dataclass
class Response(Message):
    worker_type: str
    logits: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"Response(id={self.id}, timestamp={self.timestamp},"
            f" worker_type='{self.worker_type}', logits_shape={self.logits.shape})"
        )


@dataclass
class Preemption(Event):
    pass


class Manager:
    def __init__(
        self,
        draft_queue: asyncio.Queue[Request],
        verify_queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
    ):
        print("ManagerServer: Initializing with queues")
        self.draft_queue = draft_queue
        self.verify_queue = verify_queue
        self.response_queue = response_queue
        self.pubsub = PubSub()
        print("ManagerServer: Initialized with PubSub")

    async def handle_requests(self) -> None:
        print("ManagerServer: Starting to handle requests")
        while True:
            command = (
                (await aioconsole.ainput("Enter command (draft, verify, preempt):\n"))
                .strip()
                .lower()
            )
            print(f"ManagerServer: Received command: {command}")
            if command in ["draft", "verify"]:
                # Simulating token IDs and n for the request
                tok_ids = torch.randint(0, 50000, (10,))
                n: int = 10 if command == "draft" else 100
                request = Request.create(tok_ids, n)
                print(f"ManagerServer: Enqueuing {command} task with ID {request.id}")
                if command == "draft":
                    await self.draft_queue.put(request)
                elif command == "verify":
                    await self.verify_queue.put(request)
            elif command == "preempt":
                print("ManagerServer: Preempt command received.")
                await self.preempt_all()
            else:
                print(f"ManagerServer: Invalid command received: {command}")

    async def empty_queue(self, queue: asyncio.Queue) -> None:
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                pass

    async def preempt_all(self) -> None:
        print("ManagerServer: Preempting all workers")
        # Clear the queues
        await self.empty_queue(self.draft_queue)
        await self.empty_queue(self.verify_queue)
        print("ManagerServer: Queues cleared")
        # Send preempt message to workers
        await self.pubsub.publish(Preemption.create())
        print("ManagerServer: Preempt message sent to workers")

    async def handle_responses(self) -> None:
        print("ManagerServer: Starting to handle responses")
        while True:
            response = await self.response_queue.get()
            print(f"ManagerServer: Received {response.worker_type} response {response}")
            self.response_queue.task_done()

    async def start(self) -> None:
        asyncio.create_task(self.pubsub.broadcast())
        print("ManagerServer: Started PubSub broadcast task")


class Worker:
    def __init__(
        self,
        queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
        manager: Manager,
        worker_type: str,
        gpu_id: int,
    ):
        self.manager = manager
        self.queue = queue
        self.response_queue = response_queue
        self.current_task = None
        self.worker_type = worker_type
        self.gpu_id = gpu_id
        self.model = None
        self.preempt_queue = None
        print(f"{self.worker_type}Worker: Initialized with queues")

    async def load_model(self, name: str) -> None:
        """Loads the model from the given name and moves it to the device."""
        device = cpu = "cpu"
        if torch.cuda.device_count() > self.gpu_id:
            print(f"GPU {self.gpu_id} available. Using GPU.")
            device = f"cuda:{self.gpu_id}"
        else:
            print(f"GPU {self.gpu_id} not available. Using CPU.")
        print(f"{self.worker_type}Worker: Loading model {name} on {device}")
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.model.eval()
        if device != cpu:
            print(f"{self.worker_type}Worker: Moving model to {device}")
            self.model.to(device)
        print(f"{self.worker_type}Worker: Model loaded on {device}")

    async def process_tasks(self) -> None:
        print(f"{self.worker_type}Worker: Starting to process tasks")
        self.preempt_queue = await self.manager.pubsub.subscribe()
        print(f"{self.worker_type}Worker: Subscribed to PubSub")
        while True:
            try:
                request = await self.queue.get()
                print(
                    f"{self.worker_type}Worker: Received {self.worker_type.lower()}"
                    f" task with ID {request.id}"
                )
                self.current_task = asyncio.create_task(self.perform_task(request))
                preempt_task = asyncio.create_task(self.preempt_queue.get())

                done, pending = await asyncio.wait(
                    {self.current_task, preempt_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if self.current_task in done:
                    response = self.current_task.result()
                    await self.response_queue.put(response)
                    print(
                        f"{self.worker_type}Worker: Completed"
                        f" {self.worker_type.lower()} for {response=}"
                    )
                    preempt_task.cancel()
                else:
                    preempt_message = preempt_task.result()
                    if preempt_message.timestamp > request.timestamp:
                        print(
                            f"{self.worker_type}Worker: Received preemption message"
                            f" at {preempt_message.timestamp} for task started"
                            f" at {request.timestamp}"
                        )
                        self.current_task.cancel()
                        await asyncio.wait([self.current_task])
                        print(
                            f"{self.worker_type}Worker: Task {request.id}"
                            " was preempted"
                        )
                    else:
                        print(
                            f"{self.worker_type}Worker: Ignoring outdated"
                            " preemption message"
                        )
                        continue

                self.current_task = None

            except asyncio.CancelledError:
                print(f"{self.worker_type}Worker: Process tasks cancelled")
                break

    async def perform_task(self, request: Request) -> Response:
        print(
            f"{self.worker_type}Worker: Processing"
            f" {self.worker_type.lower()} for ID {request.id}"
        )
        await asyncio.sleep(
            4 if self.worker_type == "Drafter" else 10
        )  # Simulating work
        with torch.no_grad():
            logits = torch.randn(
                request.n, len(request.tok_ids), 50000
            )  # Simulating logits
        return Response(
            id=request.id,
            timestamp=time.time(),
            worker_type=self.worker_type,
            logits=logits,
        )

    async def stop(self) -> None:
        if self.preempt_queue:
            await self.manager.pubsub.unsubscribe(self.preempt_queue)
            print(f"{self.worker_type}Worker: Unsubscribed from PubSub")


class PubSub:
    def __init__(self):
        self.queue: asyncio.Queue[Preemption] = asyncio.Queue()
        self.subscribers: Set[asyncio.Queue[Preemption]] = set()
        print("PubSub: Initialized with 0 subscribers")

    async def publish(self, message: Preemption) -> None:
        await self.queue.put(message)
        print(
            f"PubSub: Published message '{message}'. Queue size: {self.queue.qsize()}"
        )

    async def subscribe(self) -> asyncio.Queue[Preemption]:
        subscriber = asyncio.Queue()
        self.subscribers.add(subscriber)
        print(
            f"PubSub: New subscriber added. Total subscribers: {len(self.subscribers)}"
        )
        return subscriber

    async def unsubscribe(self, subscriber: asyncio.Queue[Preemption]) -> None:
        self.subscribers.remove(subscriber)
        print(f"PubSub: Subscriber removed. Total subscribers: {len(self.subscribers)}")

    async def broadcast(self) -> None:
        print("PubSub: Starting broadcast loop")
        while True:
            message = await self.queue.get()
            print(
                f"PubSub: Broadcasting message '{message}' to"
                f" {len(self.subscribers)} subscribers"
            )
            for subscriber in self.subscribers:
                await subscriber.put(message)
            print(f"PubSub: Broadcast complete. Queue size: {self.queue.qsize()}")


async def main() -> None:
    print("Main: Initializing queues")
    draft_queue = asyncio.Queue()
    verify_queue = asyncio.Queue()
    response_queue = asyncio.Queue()

    print("Main: Creating server instances")
    manager = Manager(draft_queue, verify_queue, response_queue)
    drafter = Worker(draft_queue, response_queue, manager, "Drafter", 0)
    verifier_1 = Worker(verify_queue, response_queue, manager, "Verifier", 1)
    verifier_2 = Worker(verify_queue, response_queue, manager, "Verifier", 2)

    print("Main: Loading all models")
    await asyncio.gather(
        drafter.load_model("gpt2"),
        verifier_1.load_model("gpt2"),
        verifier_2.load_model("gpt2"),
    )
    print("Main: All models loaded")

    print("Main: Starting all tasks")
    await asyncio.gather(
        manager.start(),
        manager.handle_requests(),
        manager.handle_responses(),
        drafter.process_tasks(),
        verifier_1.process_tasks(),
        verifier_2.process_tasks(),
    )


if __name__ == "__main__":
    print("Script started")
    asyncio.run(main())
