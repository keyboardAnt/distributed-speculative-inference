import asyncio
import contextlib
import gc
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple
from uuid import UUID, uuid4

import accelerate
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
                                and the remaining positions are -1.
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
        -1 in tok_ids or the end of the sequence if there is no -1s.
        Examples:
        If tok_ids = [5, 3, 2, -1, -1], n = 2, and is_draft is True, then the mask is
        [False, False, False, True, True], and if is_draft is False, then the mask is
        [False, False, True, True, False].
        If tok_ids = [5, 3, 2, -1, -1], n = 3, and is_draft is True, then the
        function raises an exception, since there are not enough empty positions in the
        sequence for generating 3 tokens. If is_draft is False, the mask is
        [False, True, True, True, False].
        If tok_ids = [5, 3, 2, 1, 0], n=2, and is_draft is True, then the function
        raises an exception (and for any n > 0), since there are no empty positions in
        the sequence. If is_draft is False, the mask is
        [False, False, False, True, True].
        """
        empty_positions = torch.nonzero(self.tok_ids[0] == -1)
        if is_draft:
            start_idx = empty_positions[0]
            if start_idx + self.n > seq_len:
                raise Exception(
                    "Not enough tokens in sequence to generate response in draft mode."
                )
            mask = torch.zeros(seq_len, dtype=bool)
            mask[start_idx : start_idx + self.n] = True
            return mask
        end_idx = seq_len if not empty_positions.any() else empty_positions[0]
        if end_idx - self.n < 0:
            raise Exception("Not enough tokens in sequence to generate response.")
        mask = torch.zeros(seq_len, dtype=bool)
        mask[end_idx + 1 - self.n : end_idx + 1] = True  # Ensure non-negative index
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
        vocab_size (int): Dimension of the output scores.
        timestamp_preemption (float): Timestamp of the last preemption.
    """

    def __init__(
        self,
        draft_queue: asyncio.Queue[Request],
        verify_queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
        tok_ids: torch.Tensor,
        max_new_tokens: int,
        vocab_size: int,
        lookahead: int,
    ):
        print("Manager: Initializing with queues")
        self.draft_queue = draft_queue
        self.verify_queue = verify_queue
        self.response_queue = response_queue
        self.seq_len = tok_ids.shape[1] + max_new_tokens
        self.tok_ids = torch.full(
            (1, self.seq_len),
            -1,
            dtype=torch.int64,
        )
        self.tok_ids[:, : tok_ids.shape[1]] = tok_ids
        self.lookahead = lookahead
        # Initialize with -1s
        self.draft_scores = torch.full(
            (1, self.seq_len, vocab_size),
            -1,
            dtype=torch.float,
        )
        self.draft_tok_ids = torch.full(
            (
                1,
                self.seq_len,
            ),
            -1,
            dtype=torch.int64,
        )
        self.id_to_mask: Dict[UUID, torch.Tensor] = {}
        self.pubsub = PubSub()
        print("Manager: Initialized with PubSub")
        self.timestamp_preemption = 0  # Initialize with 0

    async def _send(self, request: Request, queue: asyncio.Queue[Request]) -> None:
        self.id_to_mask[request.id] = request.get_mask(
            seq_len=self.seq_len, is_draft=queue == self.draft_queue
        )
        print(
            f"Manager: Enqueuing request {request.id} to {'draft' if queue == self.draft_queue else 'verify'} queue"
        )
        await queue.put(request)

    def _reset(self) -> None:
        print("Manager: Resetting draft_scores, draft_tok_ids, and id_to_mask")
        self._empty(self.draft_scores)
        self._empty(self.draft_tok_ids)
        self.id_to_mask.clear()

    @staticmethod
    def _empty(t: torch.Tensor) -> None:
        t.fill_(-1)

    # @staticmethod
    # async def _empty_queue(queue: asyncio.Queue) -> None:
    #     while not queue.empty():
    #         try:
    #             queue.get_nowait()
    #             queue.task_done()
    #         except asyncio.QueueEmpty:
    #             pass

    async def preempt_all(self) -> None:
        """
        Broadcasts a preemption message to all workers and clears the request queues.
        Updates the last preemption timestamp to the current time.

        Assumptions:
        - This method has exclusive access to modify the timestamp_preemption.

        Guarantees:
        - All workers will be notified of the preemption.
        - All request queues will be emptied.
        - The timestamp_preemption will be updated.
        """
        print("Manager: Preempting all workers")
        # Update the last preemption timestamp
        self.timestamp_preemption = time.time()
        # Send preempt message to workers
        print("Manager: Sending preempt message to workers")
        await self.pubsub.publish(Preemption.create())
        print(
            f"Manager: Preempt message sent to workers at {self.timestamp_preemption}"
        )
        # # Clear the queues
        # print("Manager: Clearing queues")
        # await self._empty_queue(self.draft_queue)
        # await self._empty_queue(self.verify_queue)
        # print("Manager: Queues cleared")

    async def run(self) -> None:
        print("Manager: Starting run")
        print(f"Manager: prompt's tok_ids.shape: {self.tok_ids.shape}")
        print(f"Manager: prompt's tok_ids: {self.tok_ids}")
        while (self.tok_ids == -1).any():  # On init, acceptance, or rejection
            print(f"Manager: number of empty tok_ids: {(self.tok_ids == -1).sum()}")
            print(f"Manager: {self.tok_ids=}")
            await self.send_requests()
            while (self.tok_ids == -1).any():  # On dropping
                print("Manager: Waiting for response")
                response: Response = await self.response_queue.get()
                print(
                    f"Manager: Received response {response}. Will process if not outdated."
                )
                if response.request_timestamp <= self.timestamp_preemption:
                    print(f"Manager: Dropping outdated response {response.id}")
                    self.response_queue.task_done()
                    continue
                print(
                    f"Manager: Processing response {response.id}. (It is not outdated.)"
                )
                # if response.id not in self.id_to_mask:
                #     print(
                #         f"Manager: Response {response.id} is not in id_to_mask. Dropping."
                #     )
                #     self.response_queue.task_done()
                #     continue
                mask: torch.Tensor = self.id_to_mask.pop(response.id)
                # print(f"Manager: Popped mask {mask} for response {response.id}")
                if response.is_draft:
                    self.draft_scores[0, mask] = response.scores
                    self.draft_tok_ids[0, mask] = response.tok_ids[
                        0, -response.scores.shape[1] :
                    ]
                    print(
                        f"Manager: Updated draft tok_ids and scores with response {response.id}. After the update, the draft tok_ids are {self.draft_tok_ids}"
                    )
                else:
                    tok_ids: torch.Tensor
                    any_rejected: bool
                    tok_ids, any_rejected = self.rejection_sampler(response, mask)
                    tok_ids_padded = torch.full_like(self.tok_ids[0, mask], -1)
                    tok_ids_padded[: len(tok_ids)] = tok_ids
                    self.tok_ids[0, mask] = tok_ids_padded
                    print(f"Manager: Updated tok_ids with response {response.id}.")
                self.response_queue.task_done()
                break
            if any_rejected:
                print(f"Manager: Rejected response {response.id}. Preempting all workers and resetting.")
                await self.preempt_all()
                self._reset()

    @torch.no_grad()
    async def send_requests(self) -> None:
        # Select n based on the number of draft tokens waiting for verification
        mask_draft_tok_ids_waiting = (self.tok_ids == -1) & (
            self.draft_tok_ids != -1
        )
        print(
            f"Manager: number of draft tokens waiting for verification: {mask_draft_tok_ids_waiting.sum()}"
        )
        n = 1 + max(0, mask_draft_tok_ids_waiting.sum())
        await self._send(
            Request.create(self.get_tok_ids_with_drafts(), n=n),
            self.verify_queue,
        )
        print(
            f"Manager: Sent verify request with n={n} and tok_ids={self.get_tok_ids_with_drafts()}"
        )
        curr_lookahead: int = min(
            self.lookahead, (self.tok_ids == -1).sum() - 1
        )
        if curr_lookahead > 0:
            await self._send(
                Request.create(self.get_tok_ids_with_drafts(), curr_lookahead),
                self.draft_queue,
            )
            print(
                f"Manager: Sent draft request with n={curr_lookahead} and tok_ids={self.get_tok_ids_with_drafts()}"
            )

    @torch.no_grad()
    def rejection_sampler(
        self, response: Response, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        print(f"Manager: Running an exact match check for response {response.id}.")
        response_len = response.tok_ids.shape[1]
        print(f"Manager: The response has length {response_len}.")
        tok_ids_accepted = response.tok_ids.clone()[0, mask[:response_len]]
        draft_tok_ids = self.draft_tok_ids[0, mask]
        mask_drafts_available = draft_tok_ids != -1
        any_rejected = (
            draft_tok_ids[mask_drafts_available]
            != tok_ids_accepted[mask_drafts_available]
        ).any()
        print(
            f"Manager: Comparing draft tok_ids {draft_tok_ids} with accepted tok_ids {tok_ids_accepted}:\n{draft_tok_ids == tok_ids_accepted}"
        )
        if any_rejected:
            idx_first_rejected = (draft_tok_ids != tok_ids_accepted).nonzero()[0].item()
            print(
                f"Manager: First rejected token is at index {idx_first_rejected}. Accepting the first {idx_first_rejected} tokens."
            )
            tok_ids_accepted = tok_ids_accepted[: idx_first_rejected + 1]
        print(
            f"Manager: Accepting new tokens. The number of accepted tokens is {len(tok_ids_accepted)}, and the tok_ids are {tok_ids_accepted}"
        )
        return tok_ids_accepted, any_rejected

    def get_tok_ids_with_drafts(self) -> torch.Tensor:
        ret: torch.Tensor = self.draft_tok_ids.clone()
        nonempty_mask = self.tok_ids != -1
        ret[nonempty_mask] = self.tok_ids[nonempty_mask]
        return ret


class Worker(ABC):
    """
    Worker (Drafters and Verifiers).

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
        timestamp_preemption (float): Timestamp of the last processed preemption.
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
        self.timestamp_preemption = 0  # Initialize with 0
        self.timestamp_request = 0  # Initialize with 0

    async def load_model(
        self,
        name: str,
        dtype: torch.dtype,
        load_in_8bit: bool,
        device_map: str,
        cache_dir: None | str = None,
    ) -> None:
        """Loads the model from the given name and moves it to the device."""
        # device = cpu = "cpu"
        # if torch.cuda.device_count() > self.gpu_id:
        #     print(f"GPU {self.gpu_id} available. Using GPU.")
        #     device = f"cuda:{self.gpu_id}"
        # else:
        #     print(f"GPU {self.gpu_id} not available. Using CPU.")
        # print(
        #     f"{self.__class__.__name__}: Loading model {name} on {device} (using device map {device_map})"
        # )
        if cache_dir is None:
            cache_dir = os.environ["TRANSFORMERS_CACHE"]
        print(f"{self.__class__.__name__}: Loading model {name} with {device_map=}")
        self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            device_map=device_map,
            cache_dir=cache_dir,
            load_in_8bit=load_in_8bit,
        )
        self.model.eval()
        # if device != cpu:
        #     print(f"{self.__class__.__name__}: Moving model to {device}")
        #     self.model.to(device)
        print(f"{self.__class__.__name__}: Model loaded")
        try:
            print(f"{self.__class__.__name__}: {self.model.device_map=}")
        except Exception as e:
            print(f"{self.__class__.__name__}: Error checking device_map: {e}")
        try:
            print(f"{self.__class__.__name__}: {self.model.hf_device_map=}")
        except Exception as e:
            print(f"{self.__class__.__name__}: Error checking hf_device_map: {e}")
        print_gpu_memory()

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
                    self.timestamp_preemption = max(
                        self.timestamp_preemption, preempt_message.timestamp
                    )
                    print(
                        f"{self.__class__.__name__}: Updated timestamp_preemption to {self.timestamp_preemption} (it is the max of the previous timestamp and the received preemption timestamp)"
                    )
                    if self.timestamp_request > self.timestamp_preemption:
                        print(
                            f"{self.__class__.__name__}: Dropping outdated preemption message. "
                            f"Last preemption timestamp: {self.timestamp_preemption}, "
                            f"last or current request timestamp: {self.timestamp_request}, "
                            f"received preemption timestamp: {preempt_message.timestamp}"
                        )
                        continue
                    print(
                        f"{self.__class__.__name__}: Processing preemption message because it was created before the last or current request. Therefore we need to terminate the current task."
                    )
                    if get_request.done():
                        print(
                            f"{self.__class__.__name__}: While receiving a preemption message, a request was received."
                        )
                        request = get_request.result()
                        print(
                            f"{self.__class__.__name__}: Received request {request.id} at timestamp {request.timestamp}"
                        )
                        if request.timestamp > self.timestamp_preemption:
                            print(
                                f"{self.__class__.__name__}: The received request {request.id} is valid (was created after the preemption). Returning it to the queue."
                            )
                            self.queue.put_nowait(request)
                            print(
                                f"{self.__class__.__name__}: Request {request.id} was returned to the queue"
                            )
                    else:
                        print(
                            f"{self.__class__.__name__}: Cancelling `get_request` to stop waiting for a queued request"
                        )
                        get_request.cancel()
                        print(f"{self.__class__.__name__}: `get_request` was cancelled")
                    if current_task is not None:
                        print(f"{self.__class__.__name__}: Cancelling current task")
                        current_task.cancel()
                        print(f"{self.__class__.__name__}: Current task was cancelled")
                    else:
                        print(f"{self.__class__.__name__}: No current task to cancel")
                    print(
                        f"{self.__class__.__name__}: Done processing preemption message"
                    )
                else:  # get_request in done
                    request = get_request.result()
                    print(
                        f"{self.__class__.__name__}: Received request with ID {request.id} at timestamp {request.timestamp}. Last preemption timestamp: {self.timestamp_preemption}"
                    )
                    if request.timestamp < self.timestamp_preemption:
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
                        self.timestamp_preemption = max(
                            self.timestamp_preemption, preempt_message.timestamp
                        )
                        print(
                            f"{self.__class__.__name__}: Updated timestamp_preemption to {self.timestamp_preemption}"
                        )

                        current_task.cancel()
                        print(f"{self.__class__.__name__}: Current task was preempted")
                    else:
                        response = current_task.result()
                        await self.response_queue.put(response)
                        print(
                            f"{self.__class__.__name__}: Task {request.id} completed. Response enqueued."
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
        Moves tensors to the model's device and back to the CPU.

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
        self.timestamp_request = request.timestamp
        print(
            f"{self.__class__.__name__}: Last or current request timestamp: {self.timestamp_request}"
        )
        print(f"{self.__class__.__name__}: Getting scores for task {request.id}")
        device = next(self.model.parameters()).device
        tok_ids = request.tok_ids.to(device)
        # Run in executor (i.e., separate thread) to avoid blocking the event loop
        scores: torch.Tensor
        tok_ids: torch.Tensor
        scores, tok_ids = await self.forward(tok_ids, request.n)
        # Move scores and tok_ids to the CPU
        scores = scores.to("cpu")
        tok_ids = tok_ids.to("cpu")
        print(f"{self.__class__.__name__}: Computed scores of shape {scores.shape}")
        return Response(
            id=request.id,
            timestamp=time.time(),
            request_timestamp=request.timestamp,
            is_draft=isinstance(self, Drafter),
            scores=scores,
            tok_ids=tok_ids,
        )

    async def forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tok_ids: An int tensor of shape (1, current_seq_len) representing the
            current prompt. All entries in tok_ids should be non-negative.
            n: The number of positions for which the return value should contain scores.
        """
        print(
            f"{self.__class__.__name__}: Using thread ID"
            f" {threading.get_native_id()} (PID: {os.getpid()})"
        )
        # only the prefix of tok_ids that is not -1 is the prompt
        tok_ids = tok_ids[:, : (tok_ids[0] != -1).sum()]
        # n = max(n, 1)  # Ensure n is at least 1
        assert n > 0, "n must be greater than 0"
        scores, sequences = self._forward(tok_ids, n)
        print(
            f"{self.__class__.__name__}: Generated sequences of shape {sequences.shape}"
        )
        return scores, sequences

    @abstractmethod
    def _forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of two tensors:
        - The scores (logits) of the generated tokens. Shape: (1, n, vocab_size)
        - The generated sequences. Shape: (1, n+current_seq_len)
        """
        pass


class Verifier(Worker):
    def _forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.model.forward(
            input_ids=tok_ids,
            attention_mask=torch.ones_like(tok_ids),
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
        )
        logits_argmax = outputs.logits.argmax(dim=-1)
        sequences = torch.cat((tok_ids[0, :1], logits_argmax[0, :])).unsqueeze(0)
        return outputs.logits, sequences


class Drafter(Worker):
    def _forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        scores = torch.stack(outputs.scores, dim=1)
        sequences = outputs.sequences
        return scores, sequences


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


def get_device_map_with_only_gpu_0(model_name):
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_name, cache_dir="/workspace/hf_cache"
        )
        max_memory = {i: 0 for i in range(1, torch.cuda.device_count())}
        max_memory[0] = f"{torch.cuda.mem_get_info(0)[0] / 1024 / 1024 / 1024:.2f} GB"
        return accelerate.infer_auto_device_map(model, max_memory=max_memory)


def setup_hf_cache():
    if torch.cuda.device_count() > 0:
        os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
        os.environ["HF_HOME"] = "/workspace/hf_cache"
    print(
        f"Main: Set Hugging Face cache directory to {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}"
    )
    print(
        f"Main: Set Hugging Face home directory to {os.environ.get('HF_HOME', 'Not set')}"
    )


async def run(
    verifier_name: str,
    drafter_name: str,
    vocab_size: int,
    verifier_dtype: torch.dtype,
    drafter_dtype: torch.dtype,
    verifier_load_in_8bit: bool,
    drafter_load_in_8bit: bool,
    lookahead: int,
    tok_ids: torch.Tensor,
    max_new_tokens: int,
) -> None:
    setup_hf_cache()
    print_gpu_memory()
    print("Main: Initializing queues")
    draft_queue = asyncio.Queue()
    verify_queue = asyncio.Queue()
    response_queue = asyncio.Queue()

    print("Main: Creating server instances")
    # Define the missing arguments
    print(f"Loading tokenizer for {verifier_name}")
    print_gpu_memory()
    print("Main: Creating manager")
    manager = Manager(
        draft_queue,
        verify_queue,
        response_queue,
        tok_ids,
        max_new_tokens,
        vocab_size,
        lookahead,
    )
    drafter = Drafter(draft_queue, response_queue, manager, 0)
    print("Main: Creating drafter")
    print_gpu_memory()
    available_gpus = torch.cuda.device_count()
    print(f"Main: Available GPUs: {available_gpus}")
    # num_verifiers = max(available_gpus - 1, 1)
    num_verifiers = 1
    print(f"Main: Number of verifiers: {num_verifiers}")
    verifiers = [
        Verifier(verify_queue, response_queue, manager, i)
        for i in range(1, num_verifiers + 1)
    ]
    print("Main: Loading all verifiers")
    await asyncio.gather(
        *[
            verifier.load_model(
                verifier_name,
                dtype=verifier_dtype,
                device_map="balanced_low_0",
                load_in_8bit=verifier_load_in_8bit,
                cache_dir=os.environ["TRANSFORMERS_CACHE"],
            )
            for verifier in verifiers
        ],
    )
    print_gpu_memory()
    print("Main: Loading drafter")
    await drafter.load_model(
        drafter_name,
        dtype=drafter_dtype,
        device_map=get_device_map_with_only_gpu_0(drafter_name),
        load_in_8bit=drafter_load_in_8bit,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    print_gpu_memory()
    print("Main: All models loaded")

    print("Main: Starting all tasks. Start measuring time NOW.")
    time_start = time.time()
    asyncio.create_task(manager.pubsub.broadcast())
    print("Main: Started PubSub broadcast task")

    # Wait for the PubSub system to be ready
    await manager.pubsub.ready.wait()
    print("Main: PubSub system is ready")
    asyncio.create_task(drafter.run())
    [asyncio.create_task(verifier.run()) for verifier in verifiers]

    # Wait for all workers to be ready
    await asyncio.gather(
        drafter.ready.wait(), *[verifier.ready.wait() for verifier in verifiers]
    )
    print("Main: All workers are ready")

    # Now start the manager
    manager_task = asyncio.create_task(manager.run())
    await manager_task
    time_end = time.time()
    print(
        f"Main: Manager task completed. Time taken: {time_end - time_start:.2f} seconds"
    )
    print(f"Main: Final tok_ids: {manager.tok_ids}")
    return manager.tok_ids


def generate(
    model_name: str,
    dtype: torch.dtype,
    load_in_8bit: bool,
    tok_ids: torch.Tensor,
    max_new_tokens: int,
) -> str:
    setup_hf_cache()
    print(f"Loading tokenizer for {model_name}")
    print_gpu_memory()
    print(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    model.eval()
    print_gpu_memory()
    # model.to("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()
    device = next(model.parameters()).device
    tok_ids = tok_ids.to(device)
    outputs = model.generate(
        input_ids=tok_ids,
        attention_mask=torch.ones_like(tok_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        return_dict_in_generate=True,
        output_scores=False,
        output_logits=False,
        output_hidden_states=False,
        output_attentions=False,
    )
    time_end = time.time()
    print(
        f"Generating with model {model_name} took {time_end - time_start:.2f} seconds"
    )
    return outputs.sequences


def garbage_collect():
    print("Collecting garbage...")
    gc.collect()
    torch.cuda.empty_cache()


def print_gpu_memory():
    print(f"The current device is {torch.cuda.current_device()}")
    for i in range(torch.cuda.device_count()):
        print(
            f"GPU {i}: {torch.cuda.mem_get_info(i)[0] / 1024 / 1024 / 1024:.2f} GB free, {torch.cuda.mem_get_info(i)[1] / 1024 / 1024 / 1024:.2f} GB total"
        )


def encode(prompt: str, tokernizer_name: str) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(tokernizer_name)
    tok_ids = tokenizer.encode(prompt, return_tensors="pt")
    del tokenizer
    garbage_collect()
    return tok_ids


def decode(tok_ids: torch.Tensor, tokernizer_name: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(tokernizer_name)
    decoded_output = tokenizer.batch_decode(tok_ids, skip_special_tokens=True)
    del tokenizer
    garbage_collect()
    return decoded_output


@torch.no_grad()
async def main():
    print("Script started")
    print_gpu_memory()
    # verifier_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    verifier_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    drafter_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    verifier_dtype: torch.dtype = torch.float16
    drafter_dtype: torch.dtype = torch.float16
    verifier_load_in_8bit: bool = True
    drafter_load_in_8bit: bool = True
    vocab_size: int = 128256
    lookahead: int = 6
    max_new_tokens: int = 100
    prompt: str = """Below is an instruction that describes a
task, paired with an input that provides
further context. Write a response that
appropriately completes the request.
### Instruction:
Produce an annotated bibliography for the provided paper.
### Input:
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: Produce an annotated bibliography for the provided paper. ### Input: The provided paper is an article about the impact of social media on mental health titled “Exploring the Impact of Social Media on Mental Health”. ### Response: Chou, S., S. C. Y., & Ioerger, M. (2019). Exploring the Impact of Social Media on Mental Health. International Journal of Mental Health, 32(2). This paper discusses the impacts of different kinds of social media on the mental health of users. Firstly, the paper uses a systematic review to explore the existing research on the link between social media use and mental health. The review provides an overview of the current state of the research and discusses both positive and negative impacts that social media can have. Results suggest that there is both a positive and a negative correlation between the use of social media and mental health. The authors then proceed to focus on the effects of four specific types of social media: Instagram, Facebook, Twitter and Youtube. For each type of social media, the authors explore the effects on three specific mental health domains: depression, anxiety, and self-esteem. The paper concludes that more research is needed to understand the relationship between social media and mental health. Bates, M., & LeMesurier, S. (2017). The effect of social media on mental health. Mental Illness, 5(1). This paper examines the potential effects of social media on aspects of mental health such as self-esteem, depression, and anxiety. The authors use an online survey of 644 participants to investigate the relationship between mental health and particular uses of Facebook and Instagram. The survey looks at different users' motives for using social media, the frequency with which they use it, feelings of loneliness or anxiety while using it, and how their real life values and perspectives are impacted by social media. Results indicate that several factors influence the relationship between social media and mental health, including positive and negative attributes of different social media types. The findings of the paper suggest that research into what might influence how users engage with social media and the particular effects of platforms and ways of using them could be beneficial to understanding the relationship between social media and mental health. Olah, Z., Z. Szatmári, & Font, S. (2019). Effects of Social Media on Mental Health. Frontiers in Psychology, 10(2). The authors of this paper explore the potential effects of social media on mental health. The paper highlights both positive and negative outcomes from the use of different types of social media. It also highlights the ways in which our mental health is inextricably linked to our social life and environment. Results from a systematic review suggest that different types of social media have different effects on individuals. For example, it found that social media use has a positive effect on collaboration, connectedness, and communication, while it can also have a negative effect on loneliness, anxiety, depression and self-esteem. The paper concludes that more research is needed to understand how these different types of social media affect our mental wellbeing.
### Response:
"""
    tok_ids = encode(prompt, verifier_name)
    # tok_ids = generate(
    #     model_name=verifier_name,
    #     dtype=verifier_dtype,
    #     load_in_8bit=verifier_load_in_8bit,
    #     tok_ids=tok_ids,
    #     max_new_tokens=max_new_tokens,
    # )
    tok_ids = await run(
        verifier_name=verifier_name,
        drafter_name=drafter_name,
        vocab_size=vocab_size,
        verifier_dtype=verifier_dtype,
        drafter_dtype=drafter_dtype,
        verifier_load_in_8bit=verifier_load_in_8bit,
        drafter_load_in_8bit=drafter_load_in_8bit,
        lookahead=lookahead,
        tok_ids=tok_ids,
        max_new_tokens=max_new_tokens,
    )
    print(f"Main: Final output: {decode(tok_ids, verifier_name)}")
    # Close all asyncio tasks or resources without waiting for them to complete
    for task in asyncio.all_tasks():
        if task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    print("Main: All servers are closed")
    print("Script completed")


if __name__ == "__main__":
    asyncio.run(main())
