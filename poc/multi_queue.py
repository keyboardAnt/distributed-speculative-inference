import asyncio
import random
import time
import uuid
from dataclasses import dataclass

import aioconsole
import torch
from transformers import AutoModelForCausalLM


@dataclass
class Request:
    task_id: str
    command: str
    timestamp: float

    @classmethod
    def create(cls, command: str):
        return cls(task_id=str(uuid.uuid4()), command=command, timestamp=time.time())


@dataclass
class Response:
    task_id: str
    result: int
    worker_type: str


@dataclass
class PreemptMessage:
    timestamp: float

    @classmethod
    def create(cls):
        return cls(timestamp=time.time())


class Manager:
    def __init__(
        self, draft_queue, verify_queue, draft_response_queue, verify_response_queue
    ):
        print("ManagerServer: Initializing with queues")
        self.draft_queue = draft_queue
        self.verify_queue = verify_queue
        self.draft_response_queue = draft_response_queue
        self.verify_response_queue = verify_response_queue
        self.pubsub = PubSub()
        print("ManagerServer: Initialized with PubSub")

    async def handle_requests(self):
        print("ManagerServer: Starting to handle requests")
        while True:
            command = (
                (await aioconsole.ainput("Enter command (draft, verify, preempt):\n"))
                .strip()
                .lower()
            )
            print(f"ManagerServer: Received command: {command}")
            if command in ["draft", "verify"]:
                request = Request.create(command)
                print(
                    f"ManagerServer: Enqueuing {command} task with ID {request.task_id}"
                )
                if command == "draft":
                    await self.draft_queue.put(request)
                elif command == "verify":
                    await self.verify_queue.put(request)
            elif command == "preempt":
                print("ManagerServer: Preempt command received, preempting all workers")
                await self.preempt_all()
            else:
                print(f"ManagerServer: Invalid command received: {command}")

    async def empty_queue(self, queue):
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                pass

    async def preempt_all(self):
        print("ManagerServer: Preempt command received, preempting all workers")
        # Clear the queues
        await self.empty_queue(self.draft_queue)
        await self.empty_queue(self.verify_queue)
        print("ManagerServer: Queues cleared")
        # Send preempt message to workers
        await self.pubsub.publish(PreemptMessage.create())
        print("ManagerServer: Preempt message sent to workers")

    async def handle_responses(self):
        print("ManagerServer: Starting to handle responses")
        while True:
            # Always check verify queue first
            if not self.verify_response_queue.empty():
                response = await self.verify_response_queue.get()
                print(f"ManagerServer: Received verify response {response}")
            elif not self.draft_response_queue.empty():
                response = await self.draft_response_queue.get()
                print(f"ManagerServer: Received draft response {response}")
            await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

    async def start(self):
        asyncio.create_task(self.pubsub.broadcast())
        print("ManagerServer: Started PubSub broadcast task")


class Worker:
    def __init__(self, queue, response_queue, manager, worker_type, gpu_id):
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

    async def process_tasks(self):
        print(f"{self.worker_type}Worker: Starting to process tasks")
        self.preempt_queue = await self.manager.pubsub.subscribe()
        print(f"{self.worker_type}Worker: Subscribed to PubSub")
        while True:
            try:
                request = await self.queue.get()
                print(
                    f"{self.worker_type}Worker: Received {self.worker_type.lower()}"
                    f" task with ID {request.task_id}"
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
                        f" {self.worker_type.lower()} for ID {response.task_id}"
                        f" with result {response.result}"
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
                            f"{self.worker_type}Worker: Task {request.task_id}"
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

    async def perform_task(self, request: Request):
        print(
            f"{self.worker_type}Worker: Processing"
            f" {self.worker_type.lower()} for ID {request.task_id}"
        )
        await asyncio.sleep(
            4 if self.worker_type == "Drafter" else 10
        )  # Simulating work
        result = (
            random.randint(100, 999)
            if self.worker_type == "Drafter"
            else random.randint(1000, 9999)
        )
        return Response(
            task_id=request.task_id, result=result, worker_type=self.worker_type
        )

    async def stop(self):
        if self.preempt_queue:
            await self.manager.pubsub.unsubscribe(self.preempt_queue)
            print(f"{self.worker_type}Worker: Unsubscribed from PubSub")


class PubSub:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.subscribers = set()
        print("PubSub: Initialized with 0 subscribers")

    async def publish(self, message):
        await self.queue.put(message)
        print(
            f"PubSub: Published message '{message}'. Queue size: {self.queue.qsize()}"
        )

    async def subscribe(self):
        subscriber = asyncio.Queue()
        self.subscribers.add(subscriber)
        print(
            f"PubSub: New subscriber added. Total subscribers: {len(self.subscribers)}"
        )
        return subscriber

    async def unsubscribe(self, subscriber):
        self.subscribers.remove(subscriber)
        print(f"PubSub: Subscriber removed. Total subscribers: {len(self.subscribers)}")

    async def broadcast(self):
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


async def main():
    print("Main: Initializing queues")
    draft_queue = asyncio.Queue()
    verify_queue = asyncio.Queue()
    draft_response_queue = asyncio.Queue()
    verify_response_queue = asyncio.Queue()

    print("Main: Creating server instances")
    manager = Manager(
        draft_queue, verify_queue, draft_response_queue, verify_response_queue
    )
    drafter = Worker(draft_queue, draft_response_queue, manager, "Drafter", 0)
    verifier_1 = Worker(verify_queue, verify_response_queue, manager, "Verifier", 1)
    verifier_2 = Worker(verify_queue, verify_response_queue, manager, "Verifier", 2)

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
