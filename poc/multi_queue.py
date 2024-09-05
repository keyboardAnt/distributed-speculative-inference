import asyncio
import time
import threading
import os
from abc import ABC, abstractmethod
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
    is_verified: bool
    scores: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"Response(id={self.id}, timestamp={self.timestamp},"
            f" is_verified={self.is_verified}, scores_shape={self.scores.shape})"
        )


@dataclass
class Preemption(Event):
    pass


class Manager:
    """
    Manages the overall system, handling requests, responses, and preemptions.

    Assumptions:
    - There is only one Manager instance in the system.
    - The Manager has exclusive access to modify the request queues.

    Guarantees:
    - Preemptions are broadcasted to all workers via PubSub.
    - Requests and responses older than the last preemption will be dropped.

    Attributes:
        draft_queue (asyncio.Queue): Queue for draft requests.
        verify_queue (asyncio.Queue): Queue for verification requests.
        response_queue (asyncio.Queue): Queue for responses from workers.
        pubsub (PubSub): PubSub system for broadcasting preemptions.
        last_preemption_timestamp (float): Timestamp of the last preemption.
    """
    def __init__(
        self,
        draft_queue: asyncio.Queue[Request],
        verify_queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
    ):
        print("Manager: Initializing with queues")
        self.draft_queue = draft_queue
        self.verify_queue = verify_queue
        self.response_queue = response_queue
        self.pubsub = PubSub()
        print("Manager: Initialized with PubSub")
        self.last_preemption_timestamp = 0  # Initialize with 0

    async def handle_requests(self) -> None:
        """
        Continuously processes responses from workers.

        Assumptions:
        - Responses are received in the order they were sent by workers.

        Guarantees:
        - Responses older than the last preemption will be dropped.
        - All valid responses will be processed in the order received.
        """
        print("Manager: Starting to handle requests")
        while True:
            command = (
                (await aioconsole.ainput("Enter command (draft, verify, preempt):\n"))
                .strip()
                .lower()
            )
            print(f"Manager: Received command: {command}")
            if command in ["draft", "verify"]:
                # Simulating token IDs and n for the request
                tok_ids = torch.tensor([[15496, 11, 616, 1438, 318]])
                n: int = 10 if command == "draft" else 1000
                request = Request.create(tok_ids, n)
                print(f"Manager: Enqueuing {command} task with ID {request.id}")
                if command == "draft":
                    await self.draft_queue.put(request)
                elif command == "verify":
                    await self.verify_queue.put(request)
            elif command == "preempt":
                print("Manager: Preempt command received.")
                await self.preempt_all()
            else:
                print(f"Manager: Invalid command received: {command}")

    async def empty_queue(self, queue: asyncio.Queue) -> None:
        while not queue.empty():
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                pass

    async def preempt_all(self) -> None:
        """
        Broadcasts a preemption message to all workers and clears the request queues.
        Updates the last preemption timestamp to the current time.

        Assumptions:
        - This method has exclusive access to modify the last_preemption_timestamp.

        Guarantees:
        - All workers will be notified of the preemption.
        - All request queues will be emptied.
        - The last_preemption_timestamp will be updated.
        """
        print("Manager: Preempting all workers")
        # Update the last preemption timestamp
        self.last_preemption_timestamp = time.time()
        # Send preempt message to workers
        print("Manager: Sending preempt message to workers")
        await self.pubsub.publish(Preemption.create())
        print(f"Manager: Preempt message sent to workers at {self.last_preemption_timestamp}")
        # Clear the queues
        print("Manager: Clearing queues")
        await self.empty_queue(self.draft_queue)
        await self.empty_queue(self.verify_queue)
        print("Manager: Queues cleared")

    async def handle_responses(self) -> None:
        """
        Continuously processes responses from workers.

        Assumptions:
        - Responses are received in the order they were sent by workers.

        Guarantees:
        - Responses older than the last preemption will be dropped.
        - All valid responses will be processed in the order received.
        """
        print("Manager: Starting to handle responses")
        while True:
            response = await self.response_queue.get()
            if response.timestamp < self.last_preemption_timestamp:
                print(f"Manager: Dropping outdated response {response.id}")
                self.response_queue.task_done()
                continue
            print(f"Manager: Received {response=}")
            # Process the response...
            self.response_queue.task_done()

    async def start(self) -> None:
        """
        Starts the PubSub broadcast loop.

        Guarantees:
        - The broadcast loop will run indefinitely until the program is terminated.
        """
        asyncio.create_task(self.pubsub.broadcast())
        print("Manager: Started PubSub broadcast task")


class Worker(ABC):
    """
    Abstract base class for workers (Drafters and Verifiers).

    Assumptions:
    - Each worker runs on a separate GPU.
    - Workers can handle preemptions at any point during task processing.
    - Workers are stateless and do not maintain any internal state between tasks.
      Therefore, a request must encapsulate all necessary information for a worker to
      process a task.

    Guarantees:
    - Only one task will be processed at a time per worker.
    - Tasks will be preempted if a valid preemption message is received.
    - Outdated tasks (older than last preemption) will not be processed.

    Attributes:
        queue (asyncio.Queue): Queue of incoming tasks.
        response_queue (asyncio.Queue): Queue for sending responses.
        manager (Manager): Reference to the manager for system-wide operations.
        gpu_id (int): ID of the GPU this worker is using.
        model: The loaded model for processing tasks.
        last_preemption_timestamp (float): Timestamp of the last processed preemption.
    """
    def __init__(
        self,
        queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
        manager: Manager,
        gpu_id: int,
    ):
        self.manager = manager
        self.queue = queue
        self.response_queue = response_queue
        self.current_task = None
        self.gpu_id = gpu_id
        self.model = None
        self.preempt_queue = None
        print(f"{self.__class__.__name__}: Initialized with queues")
        print(f"{self.__class__.__name__}: Using thread ID {threading.get_native_id()}")
        self.last_preemption_timestamp = 0  # Initialize with 0

    async def load_model(self, name: str) -> None:
        """Loads the model from the given name and moves it to the device."""
        device = cpu = "cpu"
        if torch.cuda.device_count() > self.gpu_id:
            print(f"GPU {self.gpu_id} available. Using GPU.")
            device = f"cuda:{self.gpu_id}"
        else:
            print(f"GPU {self.gpu_id} not available. Using CPU.")
        print(f"{self.__class__.__name__}: Loading model {name} on {device}")
        self.model = AutoModelForCausalLM.from_pretrained(name)
        self.model.eval()
        if device != cpu:
            print(f"{self.__class__.__name__}: Moving model to {device}")
            self.model.to(device)
        print(f"{self.__class__.__name__}: Model loaded on {device}")

    async def process_tasks(self) -> None:
        """
        Main loop for processing tasks and handling preemptions.

        Assumptions:
        - Preemption messages can be received at any time.
        - Multiple preemption messages may be received while processing a single task.

        Guarantees:
        - Tasks older than the last preemption will be dropped without processing.
        - A task will be immediately preempted if a valid preemption message is received.
        - The worker will continuously process tasks and listen for preemptions.
        - All exceptions during task processing will be caught and logged.

        Implementation details:
        - Uses asyncio.wait to simultaneously wait for task completion and preemption messages.
        - Updates last_preemption_timestamp when a valid preemption is received.
        - Cancels the current task if a valid preemption is received.
        """
        print(f"{self.__class__.__name__}: Starting to process tasks")
        self.preempt_queue = await self.manager.pubsub.subscribe()
        print(f"{self.__class__.__name__}: Subscribed to PubSub")

        while True:
            try:
                print(f"{self.__class__.__name__}: Waiting for a new task...")
                request = await self.queue.get()
                
                print(f"{self.__class__.__name__}: Received task with ID {request.id} at timestamp {request.timestamp}")
                print(f"{self.__class__.__name__}: Last preemption timestamp: {self.last_preemption_timestamp}")
                
                if request.timestamp < self.last_preemption_timestamp:
                    print(f"{self.__class__.__name__}: Dropping outdated request {request.id}")
                    self.queue.task_done()
                    continue

                print(f"{self.__class__.__name__}: Processing task with ID {request.id}")
                self.current_task = asyncio.create_task(self.perform_task(request))
                
                while not self.current_task.done():
                    preempt_task = asyncio.create_task(self.preempt_queue.get())
                    done, pending = await asyncio.wait(
                        {self.current_task, preempt_task},
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    if self.current_task in done:
                        print(f"{self.__class__.__name__}: Task {request.id} completed")
                        response = self.current_task.result()
                        await self.response_queue.put(response)
                        print(f"{self.__class__.__name__}: Response for task {request.id} enqueued")
                        break
                    else:
                        preempt_message = preempt_task.result()
                        print(f"{self.__class__.__name__}: Received preemption message at {preempt_message.timestamp}")
                        
                        if preempt_message.timestamp > request.timestamp:
                            print(f"{self.__class__.__name__}: Preempting task {request.id}")
                            self.current_task.cancel()
                            await asyncio.wait([self.current_task])
                            self.last_preemption_timestamp = preempt_message.timestamp
                            print(f"{self.__class__.__name__}: Task {request.id} was preempted")
                            print(f"{self.__class__.__name__}: Updated last_preemption_timestamp to {self.last_preemption_timestamp}")
                            break
                        else:
                            print(f"{self.__class__.__name__}: Ignoring outdated preemption message")

                print(f"{self.__class__.__name__}: Finished processing or preempting task {request.id}")
                self.current_task = None
                self.queue.task_done()

            except asyncio.CancelledError:
                print(f"{self.__class__.__name__}: Process tasks cancelled")
                break
            except Exception as e:
                print(f"{self.__class__.__name__}: Unexpected error in process_tasks: {e}")
                raise e

        print(f"{self.__class__.__name__}: Process tasks finished")

    @torch.no_grad()
    async def perform_task(self, request: Request) -> Response:
        """
        Performs the actual task processing.

        Assumptions:
        - The model is loaded and ready for inference.

        Guarantees:
        - Will return a Response object with the processing results.
        - GPU operations are performed asynchronously to avoid blocking the event loop.

        Args:
            request (Request): The request to process.

        Returns:
            Response: The processed response.
        """
        print(f"{self.__class__.__name__}: Getting scores for task {request.id}")
        device = next(self.model.parameters()).device
        tok_ids = request.tok_ids.to(device)
        loop = asyncio.get_running_loop()
        # Run in executor (i.e., separate thread) to avoid blocking the event loop
        scores: torch.Tensor = await loop.run_in_executor(
            None,
            self.get_scores,
            tok_ids,
            request.n,
        )
        print(f"{self.__class__.__name__}: Computed scores of shape {scores.shape}")
        return Response(
            id=request.id,
            timestamp=time.time(),
            is_verified=isinstance(self, Verifier),
            scores=scores,
        )

    @abstractmethod
    def get_scores(self, tok_ids: torch.Tensor, n: int) -> torch.Tensor:
        pass


class Drafter(Worker):
    def get_scores(self, tok_ids: torch.Tensor, n: int) -> torch.Tensor:
        """
        Generates up to `n` new tokens given the prompt `tok_ids`.
        Returns the scores (logits) of the generated tokens. Shape: (n, vocab_size)
        """
        print(
            f"{self.__class__.__name__}: Using thread ID"
            f" {threading.get_native_id()} (PID: {os.getpid()})"
        )
        outputs = self.model.generate(
            input_ids=tok_ids,
            attention_mask=torch.ones_like(tok_ids),
            max_new_tokens=n,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        return torch.stack(outputs.scores, dim=0).squeeze(1)


class Verifier(Worker):
    def get_scores(self, tok_ids: torch.Tensor, n: int) -> torch.Tensor:
        print(
            f"{self.__class__.__name__}: Using thread ID"
            f" {threading.get_native_id()} (PID: {os.getpid()})"
        )
        outputs = self.model.generate(
            input_ids=tok_ids,
            attention_mask=torch.ones_like(tok_ids),
            max_new_tokens=n,
            do_sample=False,
            use_cache=False,
            return_dict_in_generate=True,
            output_scores=True,
            output_logits=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        return torch.stack(outputs.scores, dim=0).squeeze(1)


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
    drafter = Drafter(draft_queue, response_queue, manager, 0)
    available_gpus = torch.cuda.device_count()
    print(f"Main: Available GPUs: {available_gpus}")
    num_verifiers = max(available_gpus - 1, 1)
    print(f"Main: Number of verifiers: {num_verifiers}")
    verifiers = [
        Verifier(verify_queue, response_queue, manager, i)
        for i in range(1, num_verifiers + 1)
    ]

    print("Main: Loading all models")
    await asyncio.gather(
        drafter.load_model("gpt2"),
        *[verifier.load_model("gpt2") for verifier in verifiers],
    )
    print("Main: All models loaded")

    print("Main: Starting all tasks")
    await asyncio.gather(
        manager.start(),
        manager.handle_requests(),
        manager.handle_responses(),
        drafter.process_tasks(),
        *[verifier.process_tasks() for verifier in verifiers],
    )


if __name__ == "__main__":
    print("Script started")
    asyncio.run(main())
