import asyncio
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
from uuid import UUID, uuid4

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    """
    Args:
        tok_ids (torch.Tensor): The token IDs of the prompt. Shape: (seq_len,).
                                The prompt populates the first part of the sequence,
                                and the remaining positions are torch.nan.
        n (int): The number of tokens to generate.
    """

    tok_ids: torch.Tensor
    n: int

    @classmethod
    def create(cls, tok_ids: torch.Tensor, n: int) -> "Request":
        return cls(id=uuid4(), timestamp=time.time(), tok_ids=tok_ids, n=n)

    def get_mask(self, seq_len: int, is_draft: bool) -> torch.Tensor:
        """
        Returns a boolean mask of shape (seq_len, ) where entries are True only at the
        positions that correspond to the response.
        If is_draft is True, the mask is True at n positions that follow the prompt (up
        to the end of the sequence).
        Otherwise, the mask is True at the n consecutive positions that end at the first
        nan in tok_ids or the end of the sequence if there is no nan.
        Examples:
        If tok_ids = [5, 3, 2, nan, nan] and n = 2, then if is_draft is True, the
        mask is [False, False, False, True, True], and if is_draft is False, the mask is
        [False, False, True, True, False].
        If tok_ids = [5, 3, 2, nan, nan] and n = 3, then if is_draft is True, the
        function raises an exception, since there are not enough tokens in the
        sequence to generate 3 tokens. If is_draft is False, the mask is
        [False, True, True, True, False].
        If tok_ids = [5, 3, 2, 1, 0] and n=2, then if is_draft is True, the function
        raises an exception (and for any n > 0), since there are not enough tokens in
        the sequence. If is_draft is False, the mask is
        [False, False, False, True, True].
        """
        nan_positions = torch.nonzero(
            torch.isnan(self.tok_ids), as_tuple=False
        ).flatten()
        if is_draft:
            if len(nan_positions) == 0:  # No NaNs, all positions are filled
                raise Exception(
                    "No space in sequence to generate response in draft mode."
                )
            start_idx = nan_positions[0]
            if start_idx + self.n > seq_len:
                raise Exception(
                    "Not enough tokens in sequence to generate response in draft mode."
                )
            mask = torch.zeros(seq_len, dtype=bool)
            mask[start_idx : start_idx + self.n] = True
        else:
            end_idx = seq_len if len(nan_positions) == 0 else nan_positions[0]
            if end_idx - self.n < 0:
                raise Exception("Not enough tokens in sequence to generate response.")
            mask = torch.zeros(seq_len, dtype=bool)
            mask[max(0, end_idx - self.n) : end_idx] = True  # Ensure non-negative index
        return mask


@dataclass
class Response(Message):
    request_timestamp: float
    is_draft: bool
    scores: torch.Tensor
    tok_ids: torch.Tensor

    def __repr__(self) -> str:
        return (
            f"Response(id={self.id}, timestamp={self.timestamp}, "
            f"request_timestamp={self.request_timestamp}, "
            f"is_draft={self.is_draft}, scores_shape={self.scores.shape}, "
            f"tok_ids={self.tok_ids})"
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
        tok_ids (torch.Tensor): Token IDs of the prompt.
        max_new_tokens (int): Maximum number of tokens to generate.
        draft_scores_dim (int): Dimension of the draft scores.
        last_preemption_timestamp (float): Timestamp of the last preemption.
    """

    def __init__(
        self,
        draft_queue: asyncio.Queue[Request],
        verify_queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
        tok_ids: torch.Tensor,
        max_new_tokens: int,
        draft_scores_dim: int,
        lookahead: int,
    ):
        print("Manager: Initializing with queues")
        self.draft_queue = draft_queue
        self.verify_queue = verify_queue
        self.response_queue = response_queue
        self.seq_len = len(tok_ids) + max_new_tokens
        self.tok_ids = torch.full(
            (self.seq_len,),
            torch.nan,
            dtype=torch.float16,
        )
        self.tok_ids[: len(tok_ids)] = tok_ids
        self.lookahead = lookahead
        # Initialize with nans
        self.draft_scores = torch.full(
            (self.seq_len, draft_scores_dim),
            torch.nan,
            dtype=torch.float32,
        )
        self.draft_tok_ids = torch.full(
            (self.seq_len,),
            torch.nan,
            dtype=torch.float16,
        )
        self.id_to_mask: Dict[UUID, torch.Tensor] = {}
        self.pubsub = PubSub()
        print("Manager: Initialized with PubSub")
        self.last_preemption_timestamp = 0  # Initialize with 0

    async def _send(self, request: Request, queue: asyncio.Queue[Request]) -> None:
        self.id_to_mask[request.id] = request.get_mask(
            seq_len=self.seq_len, is_draft=queue == self.draft_queue
        )
        await queue.put(request)

    def _reset(self) -> None:
        self._fill_nans(self.draft_scores)
        self._fill_nans(self.draft_tok_ids)
        self.id_to_mask.clear()

    @staticmethod
    def _fill_nans(t: torch.Tensor) -> None:
        t.fill_(torch.nan)

    # async def handle_requests(self) -> None:
    #     """
    #     Continuously processes responses from workers.

    #     Assumptions:
    #     - Responses are received in the order they were sent by workers.

    #     Guarantees:
    #     - Responses older than the last preemption will be dropped.
    #     - All valid responses will be processed in the order received.
    #     """
    #     print("Manager: Starting to handle requests")
    #     while True:
    #         command = (
    #             (await aioconsole.ainput("Enter command (draft, verify, preempt):\n"))
    #             .strip()
    #             .lower()
    #         )
    #         print(f"Manager: Received command: {command}")
    #         if command in ["draft", "verify"]:
    #             # Simulating token IDs and n for the request
    #             tok_ids = torch.tensor([[15496, 11, 616, 1438, 318]])
    #             n: int = 10 if command == "draft" else 100
    #             request = Request.create(tok_ids, n)
    #             print(f"Manager: Enqueuing {command} task with ID {request.id}")
    #             if command == "draft":
    #                 await self.draft_queue.put(request)
    #             elif command == "verify":
    #                 await self.verify_queue.put(request)
    #         elif command == "preempt":
    #             print("Manager: Preempt command received.")
    #             await self.preempt_all()
    #         else:
    #             print(f"Manager: Invalid command received: {command}")

    @staticmethod
    async def _empty_queue(queue: asyncio.Queue) -> None:
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
        print(
            f"Manager: Preempt message sent to workers at {self.last_preemption_timestamp}"
        )
        # Clear the queues
        print("Manager: Clearing queues")
        await self._empty_queue(self.draft_queue)
        await self._empty_queue(self.verify_queue)
        print("Manager: Queues cleared")

    # async def handle_responses(self) -> None:
    #     """
    #     Continuously processes responses from workers.

    #     Assumptions:
    #     - Responses are received in the order they were sent by workers.

    #     Guarantees:
    #     - Responses older than the last preemption will be dropped.
    #     - All valid responses will be processed in the order received.
    #     """
    #     print("Manager: Starting to handle responses")
    #     while True:
    #         response = await self.response_queue.get()
    #         if response.request_timestamp < self.last_preemption_timestamp:
    #             print(f"Manager: Dropping outdated response {response.id}")
    #             self.response_queue.task_done()
    #             continue
    #         print(f"Manager: Received {response=}")
    #         # Process the response...
    #         self.response_queue.task_done()

    async def run(self) -> None:
        print("Manager: Starting run")
        print(f"Manager: tok_ids: {self.tok_ids}")
        while self.tok_ids.isnan().any():  # On init or rejection
            print("Manager: Resetting (on init or rejection)")
            self._reset()
            await self._send(Request.create(self.tok_ids, n=1), self.verify_queue)
            print(f"Manager: Sent verify request with tok_ids={self.tok_ids} and n=1")
            curr_lookahead: int = min(self.lookahead, self.tok_ids.isnan().sum() - 1)
            print(f"Manager: The current lookahead is {curr_lookahead}")
            await self._send(
                Request.create(self.get_tok_ids_with_drafts(), curr_lookahead),
                self.draft_queue,
            )
            print(
                f"Manager: Sent draft request with tok_ids={self.get_tok_ids_with_drafts()} and n={curr_lookahead}"
            )
            while (
                self.tok_ids.isnan().any()
            ):  # continue on acceptance; stop on rejection
                print("Manager: Waiting for response")
                response: Response = await self.response_queue.get()
                print(
                    f"Manager: Received response {response}. Will process if not outdated."
                )
                if response.request_timestamp <= self.last_preemption_timestamp:
                    print(f"Manager: Dropping outdated response {response.id}")
                    self.response_queue.task_done()
                    continue
                print(f"Manager: Processing response {response}. (It is not outdated.)")
                mask: torch.Tensor = self.id_to_mask.pop(response.id)
                print(f"Manager: Popped mask {mask} for response {response.id}")
                if response.is_draft:
                    print(
                        f"Manager: Updating draft scores and tok_ids with response {response.id}"
                    )
                    self.draft_scores[mask] = response.scores
                    print(f"Manager: Updated draft scores with response {response.id}")
                    self.draft_tok_ids[mask] = response.tok_ids
                    print(f"Manager: Updated draft tok_ids with response {response.id}")

                else:
                    tok_ids: torch.Tensor
                    any_rejected: bool
                    tok_ids, any_rejected = self.rejection_sampler(response, mask)
                    self.tok_ids[mask] = tok_ids
                    print(
                        f"Manager: Updated tok_ids with response {response.id} to {tok_ids}"
                    )
                    if any_rejected:
                        print(f"Manager: Rejected response {response.id}")
                        self.response_queue.task_done()
                        break
                self.response_queue.task_done()

    def rejection_sampler(
        self, response: Response, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        # TODO: Implement rejection sampling
        raise NotImplementedError

    def get_tok_ids_with_drafts(self) -> torch.Tensor:
        ret: torch.Tensor = self.draft_tok_ids.clone()
        ret[~self.tok_ids.isnan()] = self.tok_ids[~self.tok_ids.isnan()]
        return ret


class Worker(ABC):
    """
    Abstract base class for workers (Drafters and Verifiers).

    Assumptions:
    - Each worker runs on a separate GPU if available, otherwise on CPU.
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
        ready (asyncio.Event): Event to signal when the worker is ready.
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
        self.gpu_id = gpu_id
        self.model = None
        self.ready = asyncio.Event()
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

    async def run(self) -> None:
        """
        Main loop for processing tasks and handling preemptions.

        Assumptions:
        - Preemption messages can be received at any time.
        - Multiple preemption messages may be received while processing a single task.

        Guarantees:
        - Valid preemptions are immediately processed and terminate the current task so that the GPU is released.
        - Tasks older than the last preemption will be dropped without processing.
        - The worker will continuously process tasks and listen for preemptions. In particular, computing a response must be interrupted if a preemption is received.

        Implementation:
        - We use two queues to get requests and preemptions. We wait for either one of them to complete.
        - If a preemption is received, we cancel the current task and update the last preemption timestamp. This will raise a CancelledError.
        - Otherwise (a request is received), we verify that it is valid (newer than the last preemption) and process it. The processing of the request is done in a separate thread to ensure the worker keeps listening for preemptions.
        """
        print(f"{self.__class__.__name__}: Starting to process tasks")
        self.ready.set()  # Ensure the ready event is set when run starts
        while True:
            preempt_queue = await self.manager.pubsub.subscribe(self.gpu_id)
            print(
                f"{self.__class__.__name__}: Subscribed to PubSub for GPU {self.gpu_id}"
            )
            get_request = asyncio.create_task(self.queue.get())
            get_preempt = asyncio.create_task(preempt_queue.get())
            current_task = None
            try:
                print(
                    f"{self.__class__.__name__}: Waiting for either request or preemption..."
                )
                done, pending = await asyncio.wait(
                    {get_request, get_preempt}, return_when=asyncio.FIRST_COMPLETED
                )

                if get_preempt in done:
                    preempt_message = get_preempt.result()
                    print(
                        f"{self.__class__.__name__}: Received preemption message at {preempt_message.timestamp}"
                    )
                    self.last_preemption_timestamp = max(
                        self.last_preemption_timestamp, preempt_message.timestamp
                    )
                    print(
                        f"{self.__class__.__name__}: Updated last_preemption_timestamp to {self.last_preemption_timestamp}"
                    )

                    get_request.cancel()
                    if current_task is not None:
                        print(f"{self.__class__.__name__}: Preempting current task")
                        current_task.cancel()
                        print(f"{self.__class__.__name__}: Current task was preempted")
                    else:
                        print(f"{self.__class__.__name__}: No current task to preempt")

                else:  # get_request in done
                    request = get_request.result()
                    print(
                        f"{self.__class__.__name__}: Received request with ID {request.id} at timestamp {request.timestamp}. Last preemption timestamp: {self.last_preemption_timestamp}"
                    )
                    if request.timestamp < self.last_preemption_timestamp:
                        print(
                            f"{self.__class__.__name__}: Dropping outdated request {request.id}"
                        )
                        self.queue.task_done()
                        continue

                    print(
                        f"{self.__class__.__name__}: Processing request with ID {request.id}"
                    )
                    current_task = asyncio.create_task(self.perform_task(request))
                    done, pending = await asyncio.wait(
                        {current_task, get_preempt}, return_when=asyncio.FIRST_COMPLETED
                    )

                    if get_preempt in done:
                        preempt_message = get_preempt.result()
                        print(
                            f"{self.__class__.__name__}: Received preemption message at {preempt_message.timestamp}"
                        )
                        self.last_preemption_timestamp = max(
                            self.last_preemption_timestamp, preempt_message.timestamp
                        )
                        print(
                            f"{self.__class__.__name__}: Updated last_preemption_timestamp to {self.last_preemption_timestamp}"
                        )

                        current_task.cancel()
                        print(f"{self.__class__.__name__}: Current task was preempted")
                    else:
                        response = current_task.result()
                        print(f"{self.__class__.__name__}: Task {request.id} completed")
                        await self.response_queue.put(response)
                        print(
                            f"{self.__class__.__name__}: Response for task {request.id} enqueued"
                        )

            except asyncio.CancelledError as e:
                print(f"{self.__class__.__name__}: Task {request.id} was cancelled")
                print(f"{self.__class__.__name__}: CancelledError: {e}")
                self.queue.task_done()
                raise e

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
        scores: torch.Tensor
        tok_ids: torch.Tensor
        scores, tok_ids = await loop.run_in_executor(
            None,
            self.forward,
            tok_ids,
            request.n,
        )
        print(f"{self.__class__.__name__}: Computed scores of shape {scores.shape}")
        return Response(
            id=request.id,
            timestamp=time.time(),
            request_timestamp=request.timestamp,
            is_draft=isinstance(self, Drafter),
            scores=scores,
            tok_ids=tok_ids,
        )

    @abstractmethod
    def forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pass


class Drafter(Worker):
    def forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates up to `n` new tokens given the prompt `tok_ids`.
        Returns the scores (logits) of the generated tokens. Shape: (n, vocab_size)
        """
        print(
            f"{self.__class__.__name__}: Using thread ID"
            f" {threading.get_native_id()} (PID: {os.getpid()})"
        )
        # Add the batch dimension
        tok_ids = tok_ids.unsqueeze(0)
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
        return torch.stack(outputs.scores, dim=0).squeeze(1), outputs.sequences


class Verifier(Worker):
    def forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return torch.stack(outputs.scores, dim=0).squeeze(1), outputs.sequences


class PubSub:
    def __init__(self):
        self.queue: asyncio.Queue[Preemption] = asyncio.Queue()
        self.subscribers: Dict[int, asyncio.Queue[Preemption]] = {}
        self.ready = asyncio.Event()
        print("PubSub: Initialized with 0 subscribers")

    async def publish(self, message: Preemption) -> None:
        await self.queue.put(message)
        print(
            f"PubSub: Published message '{message}'. Queue size: {self.queue.qsize()}"
        )

    async def subscribe(self, gpu_id: int) -> asyncio.Queue[Preemption]:
        if gpu_id in self.subscribers:
            print(f"PubSub: GPU {gpu_id} already subscribed. Replacing existing queue.")
            # Delete the old queue
            del self.subscribers[gpu_id]

        # Create a new queue
        subscriber = asyncio.Queue()
        self.subscribers[gpu_id] = subscriber
        print(
            f"PubSub: New subscriber added for GPU {gpu_id}."
            f" Total subscribers: {len(self.subscribers)}"
        )
        return subscriber

    async def broadcast(self) -> None:
        print("PubSub: Starting broadcast loop")
        self.ready.set()  # Signal that the broadcast loop is ready
        while True:
            message = await self.queue.get()
            print(
                f"PubSub: Broadcasting message '{message}' to"
                f" {len(self.subscribers)} subscribers"
            )
            for subscriber in self.subscribers.values():
                await subscriber.put(message)
            print(f"PubSub: Broadcast complete. Queue size: {self.queue.qsize()}")


async def main() -> None:
    print("Main: Initializing queues")
    draft_queue = asyncio.Queue()
    verify_queue = asyncio.Queue()
    response_queue = asyncio.Queue()

    print("Main: Creating server instances")
    # Define the missing arguments
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tok_ids = tokenizer.encode(
        "Hello, world! My name is ", return_tensors="pt"
    ).squeeze(0)
    max_new_tokens = 20  # Example value
    draft_scores_dim = 50257  # GPT-2 vocabulary size
    lookahead = 5  # Example value

    manager = Manager(
        draft_queue,
        verify_queue,
        response_queue,
        tok_ids,
        max_new_tokens,
        draft_scores_dim,
        lookahead,
    )
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
        drafter.load_model(model_name),
        *[verifier.load_model(model_name) for verifier in verifiers],
    )
    print("Main: All models loaded")

    print("Main: Starting all tasks")
    broadcast_task = asyncio.create_task(manager.pubsub.broadcast())
    print("Main: Started PubSub broadcast task")

    # Wait for the PubSub system to be ready
    await manager.pubsub.ready.wait()
    print("Main: PubSub system is ready")

    # Start all worker tasks
    worker_tasks = [
        asyncio.create_task(drafter.run()),
        *[asyncio.create_task(verifier.run()) for verifier in verifiers],
    ]

    # Wait for all workers to be ready
    await asyncio.gather(
        drafter.ready.wait(), *[verifier.ready.wait() for verifier in verifiers]
    )
    print("Main: All workers are ready")

    # Now start the manager
    manager_task = asyncio.create_task(manager.run())

    # Wait for all tasks to complete
    await asyncio.gather(manager_task, *worker_tasks, broadcast_task)


if __name__ == "__main__":
    print("Script started")
    asyncio.run(main())
