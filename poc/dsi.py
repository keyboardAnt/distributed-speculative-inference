import asyncio
import contextlib
from functools import cache
import gc
import os
import threading
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Tuple, Type
from uuid import UUID, uuid4
from datetime import datetime

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
            f"tok_ids:\n{self.tok_ids})"
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
        print(f"{self.__class__.__name__}: Initializing with queues")
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
        self.requested_verify = torch.full_like(self.draft_tok_ids, False, dtype=torch.bool)
        self.requested_draft = self.requested_verify.clone()
        self.pubsub = PubSub()
        print(f"{self.__class__.__name__}: Initialized with PubSub")

    async def _send(self, request: Request, queue: asyncio.Queue[Request]) -> None:
        self.id_to_mask[request.id] = request.get_mask(
            seq_len=self.seq_len, is_draft=queue == self.draft_queue
        )
        requested = self.requested_verify if queue == self.verify_queue else self.requested_draft
        if requested[0, self.id_to_mask[request.id]].all():
            print(
                f"{self.__class__.__name__}: Won't send {('verify' if queue == self.verify_queue else 'draft')} request {request.id} because it covers already requested positions."
            )
            return
        requested[0, self.id_to_mask[request.id]] = True
        print(
            f"{self.__class__.__name__}: Enqueuing request {request.id} to {'draft' if queue == self.draft_queue else 'verify'} queue"
        )
        await queue.put(request)
        print(
            f"{self.__class__.__name__}: Sent {('verify' if queue == self.verify_queue else 'draft')} request with n={request.n} and tok_ids={self.get_tok_ids_with_drafts()}"
        )


    def _reset(self) -> None:
        print(f"{self.__class__.__name__}: Resetting draft_scores, draft_tok_ids, and id_to_mask")
        self.draft_scores.fill_(-1)
        self.draft_tok_ids.fill_(-1)
        self.id_to_mask.clear()
        self.requested_verify.fill_(False)
        self.requested_draft.fill_(False)

    async def preempt_all(self) -> None:
        """
        Broadcasts a preemption message to all workers and clears the request queues.

        Assumptions:
        - This method has exclusive access to modify the timestamp_preemption.

        Guarantees:
        - All workers will be notified of the preemption.
        - All request queues will be emptied.
        """
        print(f"{self.__class__.__name__}: Preempting all workers")
        # Send preempt message to workers
        print(f"{self.__class__.__name__}: Sending preempt message to workers")
        await self.pubsub.publish(Preemption.create())
        print(
            f"{self.__class__.__name__}: Preempt message sent to workers"
        )
        # # Clear the queues
        # print(f"{self.__class__.__name__}: Clearing queues")
        # await self._empty_queue(self.draft_queue)
        # await self._empty_queue(self.verify_queue)
        # print(f"{self.__class__.__name__}: Queues cleared")

    async def run(self) -> None:
        print(f"{self.__class__.__name__}: Starting run")
        print(f"{self.__class__.__name__}: prompt's tok_ids.shape: {self.tok_ids.shape}")
        print(f"{self.__class__.__name__}: prompt's tok_ids: {self.tok_ids}")
        to_verify_semaphore: int = self.verify_queue.maxsize
        print(f"{self.__class__.__name__}: {to_verify_semaphore=}")
        to_draft: bool = True
        while (self.tok_ids == -1).any():  # On init, acceptance, or rejection
            print(f"{self.__class__.__name__}: number of empty tok_ids: {(self.tok_ids == -1).sum()}")
            print(f"{self.__class__.__name__}: {self.tok_ids=}")
            any_rejected: bool = False
            if to_verify_semaphore > 0:
                await self.send_reqeust_verify()
                to_verify_semaphore -= 1
                print(f"{self.__class__.__name__}: {to_verify_semaphore=}")
            if to_draft:
                await self.send_request_draft()
            to_draft = False
            while (self.tok_ids == -1).any():  # On dropping
                print(f"{self.__class__.__name__}: Waiting for response")
                response: Response = await self.response_queue.get()
                print(
                    f"{self.__class__.__name__}: Received response {response}. Will process if not outdated."
                )
                if response.is_draft:
                    to_draft = True
                else:
                    to_verify_semaphore += 1
                    print(f"{self.__class__.__name__}: {to_verify_semaphore=}")
                if response.id not in self.id_to_mask:
                    print(
                        f"{self.__class__.__name__}: Response {response.id} is not in id_to_mask. Dropping."
                    )
                    self.response_queue.task_done()
                    if to_draft or (to_verify_semaphore > 0):
                        print(f"{self.__class__.__name__}: Breaking out the listening loop because is a request to send. ({to_draft=}, {to_verify_semaphore=})")
                        break
                    continue
                print(
                    f"{self.__class__.__name__}: Processing response {response.id}. (It is not outdated.)"
                )
                mask: torch.Tensor = self.id_to_mask.pop(response.id)
                if response.is_draft:
                    self.draft_scores[0, mask] = response.scores
                    self.draft_tok_ids[0, mask] = response.tok_ids[
                        0, -response.scores.shape[1] :
                    ]
                    print(
                        f"{self.__class__.__name__}: Updated draft tok_ids and scores with response {response.id}. After the update, the draft tok_ids are {self.draft_tok_ids}"
                    )
                    mask_verified = self.tok_ids[0, mask] != -1
                    if (self.tok_ids[0, mask][mask_verified] != self.draft_tok_ids[0, mask][mask_verified]).any():
                        print(f"{self.__class__.__name__}: The draft response {response.id} covers positions that were already verified. The draft token ids differ from the verified ones. (Draft tok_ids: {self.draft_tok_ids[0, mask]}, verified tok_ids: {self.tok_ids[0, mask]})")
                        any_rejected = True
                        self.response_queue.task_done()
                        break
                else:
                    tok_ids, any_rejected = self.rejection_sampler(response, mask)
                    tok_ids_padded = torch.full_like(self.tok_ids[0, mask], -1)
                    tok_ids_padded[: len(tok_ids)] = tok_ids
                    self.tok_ids[0, mask] = tok_ids_padded
                self.response_queue.task_done()
                break
            if any_rejected:
                print(f"{self.__class__.__name__}: Rejected response {response.id}. Preempting all workers and resetting.")
                await self.preempt_all()
                self._reset()
                to_draft = True

    @torch.no_grad()
    async def send_reqeust_verify(self) -> None:
        # Select n based on the number of draft tokens waiting for verification
        mask_draft_tok_ids_to_verify = (self.tok_ids == -1) & (
            self.draft_tok_ids != -1
        )
        print(
            f"{self.__class__.__name__}: number of draft tokens waiting for verification: {mask_draft_tok_ids_to_verify.sum()}"
        )
        n = 1 + max(0, mask_draft_tok_ids_to_verify.sum())
        await self._send(
            Request.create(self.get_tok_ids_with_drafts(), n=n),
            self.verify_queue,
        )

    @torch.no_grad()
    async def send_request_draft(self) -> None:
        mask_draft_tok_ids_to_draft = (self.tok_ids == -1) & (self.draft_tok_ids == -1)
        curr_lookahead: int = min(
            self.lookahead, mask_draft_tok_ids_to_draft.sum() - 1
        )
        if curr_lookahead > 0:
            await self._send(
                Request.create(self.get_tok_ids_with_drafts(), curr_lookahead),
                self.draft_queue,
            )

    @torch.no_grad()
    def rejection_sampler(
        self, response: Response, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, bool]:
        print(f"{self.__class__.__name__}: Running an exact match check for response {response.id}.")
        response_len = response.tok_ids.shape[1]
        print(f"{self.__class__.__name__}: The response has length {response_len}.")
        tok_ids_accepted = response.tok_ids.clone()[0, mask[:response_len]]
        draft_tok_ids = self.draft_tok_ids[0, mask]
        mask_drafts_available = draft_tok_ids != -1
        any_rejected = (
            draft_tok_ids[mask_drafts_available]
            != tok_ids_accepted[mask_drafts_available]
        ).any()
        print(
            f"{self.__class__.__name__}: Comparing draft tok_ids {draft_tok_ids} with accepted tok_ids {tok_ids_accepted}:\n{draft_tok_ids == tok_ids_accepted}"
        )
        if any_rejected:
            idx_first_rejected = (draft_tok_ids != tok_ids_accepted).nonzero()[0].item()
            print(
                f"{self.__class__.__name__}: First rejected token is at index {idx_first_rejected}. Accepting the first {idx_first_rejected} tokens."
            )
            tok_ids_accepted = tok_ids_accepted[: idx_first_rejected + 1]
        print(
            f"{self.__class__.__name__}: Accepting new tokens. The number of accepted tokens is {len(tok_ids_accepted)}, and the tok_ids are {tok_ids_accepted}"
        )
        return tok_ids_accepted, any_rejected

    def get_tok_ids_with_drafts(self) -> torch.Tensor:
        ret: torch.Tensor = self.draft_tok_ids.clone()
        nonempty_mask = self.tok_ids != -1
        ret[nonempty_mask] = self.tok_ids[nonempty_mask]
        return ret


class ManagerSequential(Manager):
    async def run(self) -> None:
        print(f"{self.__class__.__name__}: Starting run")
        print(f"{self.__class__.__name__}: prompt's tok_ids.shape: {self.tok_ids.shape}")
        print(f"{self.__class__.__name__}: prompt's tok_ids: {self.tok_ids}")
        while (self.tok_ids == -1).any():  # On init, acceptance, or rejection
            print(f"{self.__class__.__name__}: number of empty tok_ids: {(self.tok_ids == -1).sum()}")
            print(f"{self.__class__.__name__}: {self.tok_ids=}")
            # 1. Draft
            mask_draft_tok_ids_to_draft = (self.tok_ids == -1) & (self.draft_tok_ids == -1)
            curr_lookahead: int = min(
                self.lookahead, mask_draft_tok_ids_to_draft.sum() - 1
            )
            if curr_lookahead > 0:
                await self._send(
                    Request.create(self.get_tok_ids_with_drafts(), curr_lookahead),
                    self.draft_queue,
                )
                print(f"{self.__class__.__name__}: Waiting for draft response")
                response_draft: Response = await self.response_queue.get()
                print(
                    f"{self.__class__.__name__}: Received draft response {response_draft}."
                )
                mask: torch.Tensor = self.id_to_mask.pop(response_draft.id)
                self.draft_scores[0, mask] = response_draft.scores
                self.draft_tok_ids[0, mask] = response_draft.tok_ids[
                    0, -response_draft.scores.shape[1] :
                ]
                print(
                    f"{self.__class__.__name__}: Updated draft tok_ids and scores with response {response_draft.id}. After the update, the draft tok_ids are {self.draft_tok_ids}"
                )
                self.response_queue.task_done()
            # 2. Verify
            mask_draft_tok_ids_to_verify = (self.tok_ids == -1) & (self.draft_tok_ids != -1)
            print(
                f"{self.__class__.__name__}: number of draft tokens waiting for verification: {mask_draft_tok_ids_to_verify.sum()}"
            )
            n = 1 + max(0, mask_draft_tok_ids_to_verify.sum())
            await self._send(
                Request.create(self.get_tok_ids_with_drafts(), n=n),
                self.verify_queue,
            )
            response_verify: Response = await self.response_queue.get()
            print(
                f"{self.__class__.__name__}: Received verify response {response_verify}."
            )
            mask: torch.Tensor = self.id_to_mask.pop(response_verify.id)
            tok_ids: torch.Tensor
            any_rejected: bool
            tok_ids, any_rejected = self.rejection_sampler(response_verify, mask)
            tok_ids_padded = torch.full_like(self.tok_ids[0, mask], -1)
            tok_ids_padded[: len(tok_ids)] = tok_ids
            self.tok_ids[0, mask] = tok_ids_padded
            self.response_queue.task_done()
            if any_rejected:
                print(f"{self.__class__.__name__}: Rejected verify response {response_verify.id}.")
                self._reset()

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
        worker_id (int): ID of this worker.
        model: The loaded model for processing tasks.
        timestamp_preemption (float): Timestamp of the last processed preemption.
        ready (asyncio.Event): Event to signal when the worker is ready.
    """

    def __init__(
        self,
        queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
        manager: Manager,
        worker_id: int,
    ):
        self.manager = manager
        self.queue = queue
        self.response_queue = response_queue
        self.worker_id = worker_id
        self.model = None
        self.ready = asyncio.Event()
        print(f"{self.__class__.__name__}: Initialized with queues")
        print(f"{self.__class__.__name__}: Using thread ID {threading.get_native_id()}")
        self.timestamp_preemption = 0  # Initialize with 0
        self.timestamp_request = 0  # Initialize with 0

    def reset(self) -> None:
        self.timestamp_preemption = 0
        self.timestamp_request = 0
        print(f"{self.__class__.__name__}: Resetting timestamp_preemption and timestamp_request")

    async def load_model(
        self,
        name: str,
        dtype: torch.dtype,
        load_in_8bit: bool,
        device_map: None | str,
        cache_dir: None | str = None,
    ) -> None:
        """Loads the model from the given name and moves it to the device."""
        # device = cpu = "cpu"
        # if torch.cuda.device_count() > self.worker_id:
        #     print(f"GPU {self.worker_id} available. Using GPU.")
        #     device = f"cuda:{self.worker_id}"
        # else:
        #     print(f"GPU {self.worker_id} not available. Using CPU.")
        # print(
        #     f"{self.__class__.__name__}: Loading model {name} on {device} (using device map {device_map})"
        # )
        if cache_dir is None:
            cache_dir = os.environ["TRANSFORMERS_CACHE"]
        if device_map is None:
            print(f"{self.__class__.__name__}: Loading model {name} without specifying device map")
            self.model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            load_in_8bit=load_in_8bit,
        )
        else:
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
        print(f"{self.__class__.__name__}: {next(self.model.parameters()).device=}")


    async def cancel_task(self, task: asyncio.Task) -> None:
        print(f"{self.__class__.__name__} ({self.worker_id}): Cancelling task: {task}")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print(f"{self.__class__.__name__} ({self.worker_id}): Task was cancelled")
            return
        except Exception as e:
            print(f"{self.__class__.__name__} ({self.worker_id}): Task had an exception: {e}")
            raise

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
        print(f"{self.__class__.__name__} ({self.worker_id}): Starting to process tasks")
        self.ready.set()  # Ensure the ready event is set when run starts
        while True:
            preempt_queue = await self.manager.pubsub.subscribe(self.worker_id)
            print(
                f"{self.__class__.__name__} ({self.worker_id}): Subscribed to PubSub for GPU {self.worker_id}"
            )
            get_request = asyncio.create_task(self.queue.get())
            get_preempt = asyncio.create_task(preempt_queue.get())
            current_task = None
            print(
                f"{self.__class__.__name__} ({self.worker_id}): Waiting for either request or preemption..."
            )
            done, pending = await asyncio.wait(
                {get_request, get_preempt}, return_when=asyncio.FIRST_COMPLETED
            )

            if get_preempt in done:
                preempt_message = get_preempt.result()
                print(
                    f"{self.__class__.__name__} ({self.worker_id}): Received preemption message at {preempt_message.timestamp}"
                )
                self.timestamp_preemption = max(
                    self.timestamp_preemption, preempt_message.timestamp
                )
                print(
                    f"{self.__class__.__name__} ({self.worker_id}): Updated timestamp_preemption to {self.timestamp_preemption} (it is the max of the previous timestamp and the received preemption timestamp)"
                )
                if self.timestamp_request > self.timestamp_preemption:
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Dropping outdated preemption message. "
                        f"Last preemption timestamp: {self.timestamp_preemption}, "
                        f"last or current request timestamp: {self.timestamp_request}, "
                        f"received preemption timestamp: {preempt_message.timestamp}"
                    )
                    continue
                print(
                    f"{self.__class__.__name__} ({self.worker_id}): Processing preemption message because it was created before the last or current request. Therefore we need to terminate the current task."
                )
                if get_request.done():
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): While receiving a preemption message, a request was received."
                    )
                    request = get_request.result()
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Received request {request.id} at timestamp {request.timestamp}"
                    )
                    if request.timestamp > self.timestamp_preemption:
                        print(
                            f"{self.__class__.__name__} ({self.worker_id}): The received request {request.id} is valid (was created after the preemption). Returning it to the queue."
                        )
                        self.queue.put_nowait(request)
                        print(
                            f"{self.__class__.__name__} ({self.worker_id}): Request {request.id} was returned to the queue"
                        )
                else:
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Cancelling `get_request` to stop waiting for a queued request"
                    )
                    await self.cancel_task(get_request)
                if current_task is not None:
                    await self.cancel_task(current_task)
                    self.queue.task_done()
                    print(f"{self.__class__.__name__} ({self.worker_id}): Current task was preempted")
                else:
                    print(f"{self.__class__.__name__} ({self.worker_id}): No current task to cancel")
                print(
                    f"{self.__class__.__name__} ({self.worker_id}): Done processing preemption message"
                )
            else:  # get_request in done
                request = get_request.result()
                print(
                    f"{self.__class__.__name__} ({self.worker_id}): Received request with ID {request.id} at timestamp {request.timestamp}. Last preemption timestamp: {self.timestamp_preemption}"
                )
                if request.timestamp < self.timestamp_preemption:
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Dropping outdated request {request.id}"
                    )
                    self.queue.task_done()
                    continue

                print(
                    f"{self.__class__.__name__} ({self.worker_id}): Processing request with ID {request.id}"
                )
                current_task = asyncio.create_task(self.perform_task(request))
                done, pending = await asyncio.wait(
                    {current_task, get_preempt}, return_when=asyncio.FIRST_COMPLETED
                )

                if get_preempt in done:
                    preempt_message = get_preempt.result()
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Received preemption message at {preempt_message.timestamp}"
                    )
                    self.timestamp_preemption = max(
                        self.timestamp_preemption, preempt_message.timestamp
                    )
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Updated timestamp_preemption to {self.timestamp_preemption}"
                    )
                    await self.cancel_task(current_task)
                    self.queue.task_done()
                    print(f"{self.__class__.__name__} ({self.worker_id}): Current task was preempted")
                else:
                    response = current_task.result()
                    await self.response_queue.put(response)
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Task {request.id} completed. Response enqueued."
                    )
            for task in pending:
                await self.cancel_task(task)
            print(f"{self.__class__.__name__} ({self.worker_id}): Cancelled pending tasks")

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
            f"{self.__class__.__name__} ({self.worker_id}): Last or current request timestamp: {self.timestamp_request}"
        )
        print(f"{self.__class__.__name__} ({self.worker_id}): Getting scores for task {request.id}")
        device = next(self.model.parameters()).device
        tok_ids = request.tok_ids.to(device)
        # Run in executor (i.e., separate thread) to avoid blocking the event loop
        scores: torch.Tensor
        tok_ids: torch.Tensor
        scores, tok_ids = await self.forward(tok_ids, request.n)
        # Move scores and tok_ids to the CPU
        scores = scores.to("cpu")
        tok_ids = tok_ids.to("cpu")
        print(f"{self.__class__.__name__} ({self.worker_id}): Computed scores of shape {scores.shape}")
        return Response(
            id=request.id,
            timestamp=time.time(),
            request_timestamp=request.timestamp,
            is_draft=isinstance(self, Drafter),
            scores=scores,
            tok_ids=tok_ids,
        )

    @torch.no_grad()
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
            f"{self.__class__.__name__} ({self.worker_id}): Using thread ID"
            f" {threading.get_native_id()} (PID: {os.getpid()})"
        )
        # only the prefix of tok_ids that is not -1 is the prompt
        tok_ids = tok_ids[:, : (tok_ids[0] != -1).sum()]
        # n = max(n, 1)  # Ensure n is at least 1
        assert n > 0, "n must be greater than 0"
        scores, sequences = await self._forward(tok_ids, n)
        print(
            f"{self.__class__.__name__} ({self.worker_id}): Generated sequences of shape {sequences.shape}"
        )
        return scores, sequences

    @abstractmethod
    async def _forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple of two tensors:
        - The scores (logits) of the generated tokens. Shape: (1, n, vocab_size)
        - The generated sequences. Shape: (1, n+current_seq_len)
        """
        pass


class Verifier(Worker):
    @torch.no_grad()
    async def _forward(
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
        if n > 1:
            tok_ids = tok_ids[:, :-n+1]
        sequences = torch.cat((tok_ids[0, :], logits_argmax[0, -n:])).unsqueeze(0)
        return outputs.logits, sequences


class VerifierSlow(Verifier):
    @torch.no_grad()
    async def _forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        await asyncio.sleep(5)
        return await super()._forward(tok_ids, n)


class Drafter(Worker):
    @torch.no_grad()
    async def _forward(
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

class DrafterOracle(Drafter):
    async def _forward(
        self, tok_ids: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Llama 3.1 8B 8bit
        # oracle_tok_ids = torch.tensor([[128000,  39314,    374,    459,   7754,    430,  16964,    264,   3465,
        #      11,  35526,    449,    459,   1988,    430,   5825,   4726,   2317,
        #      13,   9842,    264,   2077,    430,  36001,  45695,    279,   1715,
        #     627,  14711,  30151,    512,  23340,    279,   1495,   1139,   1403,
        #   20406,  43743,    627,  14711,   5688,    512,    791,  16659,    649,
        #     387,   3974,    477,   4200,     11,    719,    449,    279,  28522,
        #   14691,    304,   1690,   5596,    315,    279,   1917,     11,   1690,
        #    5220,    690,   3469,    369,   4200,  16659,    304,   2015,    311,
        #   30437,    279,   9041,    315,  17563,     13,  21382,  16659,   1101,
        #    4546,    459,   1358,    315,  22934,   1093,   1694,   3025,    311,
        #    4667,    449,   1274,    304,   2204,   5596,    315,    279,   1917,
        #     627,  14711,   6075,     25,    720,    791,  16659,    649,    387,
        #    3974,    477,   4200,     11,    719,    449,    279,  28522,  14691,
        #     304,   1690,   5596,    315,    279,   1917,     11,   1690,   5220,
        #     690,   3469,    369,   4200,  16659,    304,   2015,    311,  30437,
        #     279,   9041,    315,  17563,     13,  21382,  16659,   1101,   4546,
        #     459,   1358,    315,  22934,   1093,   1694,   3025,    311,   4667,
        #     449,   1274,    304,   2204,   5596,    315,    279,   1917,     13,
        #     720,  34126,  16659,   1101,   4546,    459,   1358,    315,  22934,
        #    1093,   1694,   3025,    311,   4667,    449,   1274,    304,   2204,
        #    5596,    315,    279,   1917,     13,    578,  16659,    649,    387,
        #    3974,    477,   4200,     11,    719,    449,    279,  28522,  14691,
        #     304,   1690,   5596,    315,    279]])
        # Llama 3.1 70B 8bit
        oracle_tok_ids = torch.tensor([[128000,  39314,    374,    459,   7754,    430,  16964,    264,   3465,
             11,  35526,    449,    459,   1988,    430,   5825,   4726,   2317,
             13,   9842,    264,   2077,    430,  36001,  45695,    279,   1715,
            627,  14711,  30151,    512,  23340,    279,   1495,   1139,   1403,
          20406,  43743,    627,  14711,   5688,    512,    791,  16659,    649,
            387,   3974,    477,   4200,     11,    719,    449,    279,  28522,
          14691,    304,   1690,   5596,    315,    279,   1917,     11,   1690,
           5220,    690,   3469,    369,   4200,  16659,    304,   2015,    311,
          30437,    279,   9041,    315,  17563,     13,  21382,  16659,   1101,
           4546,    459,   1358,    315,  22934,   1093,   1694,   3025,    311,
           4667,    449,   1274,    304,   2204,   5596,    315,    279,   1917,
            627,  14711,   6075,     25,   4815,    791,  16659,    649,    387,
           3974,    477,   4200,     11,    719,    449,    279,  28522,  14691,
            304,   1690,   5596,    315,    279,   1917,     11,   1690,   5220,
            690,   3469,    369,   4200,  16659,    304,   2015,    311,  30437,
            279,   9041,    315,  17563,    382,  34126,  16659,   1101,   4546,
            459,   1358,    315,  22934,   1093,   1694,   3025,    311,   4667,
            449,   1274,    304,   2204,   5596,    315,    279,   1917,     13,
         128009, 128006,  78191, 128007,    271,    791,  16659,    649,    387,
           3974,    477,   4200,     11,    719,    449,    279,  28522,  14691,
            304,   1690,   5596,    315,    279,   1917,     11,   1690,   5220,
            690,   3469,    369,   4200,  16659,    304,   2015,    311,  30437,
            279,   9041,    315,  17563,    382]])
        idx_first_new_token = tok_ids.shape[1]
        ret_tok_ids = oracle_tok_ids[:, idx_first_new_token:idx_first_new_token+n]
        ret_scores = torch.zeros((1, n, self.model.config.vocab_size))
        return ret_scores, ret_tok_ids

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

    async def subscribe(self, worker_id: int) -> asyncio.Queue[Preemption]:
        if worker_id in self.subscribers:
            print(f"PubSub: GPU {worker_id} already subscribed. Replacing existing queue.")
            # Delete the old queue
            del self.subscribers[worker_id]

        # Create a new queue
        subscriber = asyncio.Queue()
        self.subscribers[worker_id] = subscriber
        print(
            f"PubSub: New subscriber added for GPU {worker_id}."
            f" Total subscribers: {len(self.subscribers)}"
        )
        return subscriber

    async def broadcast(self) -> None:
        print("PubSub: Starting broadcast loop")
        self.ready.set()  # Signal that the broadcast loop is ready
        while True:
            try:
                message = await self.queue.get()
                print(
                    f"PubSub: Broadcasting message '{message}' to"
                    f" {len(self.subscribers)} subscribers"
                )
                for subscriber in self.subscribers.values():
                    await subscriber.put(message)
                print(f"PubSub: Broadcast complete. Queue size: {self.queue.qsize()}")
            except asyncio.CancelledError:
                print("PubSub: Broadcast task was cancelled")
                raise
            except Exception as e:
                print(f"PubSub: Exception in broadcast loop: {e}")


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


def load_device_map(file_name):
    with open(file_name, "r") as f:
        device_map = json.load(f)
    return device_map


async def setup_workers_and_pubsub(
        verify_queue: asyncio.Queue,
        draft_queue: asyncio.Queue,
        response_queue: asyncio.Queue,
        manager: Manager,
        verifier_cls: Type[Verifier],
        drafter_cls: Type[Drafter],
        verifier_name: str,
        drafter_name: str,
        verifier_dtype: torch.dtype,
        drafter_dtype: torch.dtype,
        verifier_load_in_8bit: bool,
        drafter_load_in_8bit: bool,
        num_verifiers: int,
        verifiers_device_maps: list[dict],
        drafter_device_map: dict | None,
) -> None:
    setup_hf_cache()
    print_gpu_memory()
    print("Main: Creating server instances")
    drafter = drafter_cls(queue=draft_queue, response_queue=response_queue, manager=manager, worker_id=0)
    print("Main: Created drafter")
    print_gpu_memory()
    available_gpus = torch.cuda.device_count()
    print(f"Main: Available GPUs: {available_gpus}")
    verifiers = [
        verifier_cls(queue=verify_queue, response_queue=response_queue, manager=manager, worker_id=i)
        for i in range(1, num_verifiers + 1)
    ]
    print("Main: Loading all verifiers")
    await asyncio.gather(
        *[
            verifier.load_model(
                verifier_name,
                dtype=verifier_dtype,
                # device_map="auto",
                # device_map="balanced_low_0",
                device_map=device_map,
                load_in_8bit=verifier_load_in_8bit,
                cache_dir=os.environ["TRANSFORMERS_CACHE"],
            )
            for verifier, device_map in zip(verifiers, verifiers_device_maps)
        ],
    )
    print_gpu_memory()
    print("Main: Loading drafter")
    await drafter.load_model(
        drafter_name,
        dtype=drafter_dtype,
        # device_map=None,
        device_map=drafter_device_map,
        load_in_8bit=drafter_load_in_8bit,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    print_gpu_memory()
    print("Main: All models loaded")
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


async def run(
    manager_cls: Type[Manager],
    verifier_cls: Type[Verifier],
    drafter_cls: Type[Drafter],
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
    print_gpu_memory()
    num_verifiers = 2
    print(f"Main: Number of verifiers: {num_verifiers}")
    draft_queue = asyncio.Queue(maxsize=1)
    verify_queue = asyncio.Queue(maxsize=num_verifiers)
    response_queue = asyncio.Queue()
    print("Main: Queues created")
    manager = manager_cls(
        draft_queue,
        verify_queue,
        response_queue,
        tok_ids,
        max_new_tokens,
        vocab_size,
        lookahead,
    )
    print(f"Main: Created {manager.__class__.__name__}")
    verifier_device_map = load_device_map("/workspace/distributed-speculative-inference/poc/device_map_meta-llama_Meta-Llama-3.1-70B-Instruct_8bit_on_3A40_custom.json")
    verifier_2_device_map = {k: v + 3 for k, v in verifier_device_map.items()}
    verifiers_device_maps = [verifier_device_map, verifier_2_device_map]
    for i, device_map in enumerate(verifiers_device_maps):
        print(f"Main: Verifier {i} device map: {device_map}")
    await setup_workers_and_pubsub(
        verify_queue=verify_queue,
        draft_queue=draft_queue,
        response_queue=response_queue,
        manager=manager,
        verifier_cls=verifier_cls,
        drafter_cls=drafter_cls,
        verifier_name=verifier_name,
        drafter_name=drafter_name,
        verifier_dtype=verifier_dtype,
        drafter_dtype=drafter_dtype,
        verifier_load_in_8bit=verifier_load_in_8bit,
        drafter_load_in_8bit=drafter_load_in_8bit,
        num_verifiers=num_verifiers,
        verifiers_device_maps=verifiers_device_maps,
        drafter_device_map=None,
    )
    print("Main: Start measuring time NOW and run manager.")
    time_start = time.time()
    await manager.run()
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
    prompt: str = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
Break the text into two logical paragraphs.
### Input:
The meetings can be live or virtual, but with the pandemic continuing in many parts of the world, many companies will opt for virtual meetings in order to minimize the spread of illness. Virtual meetings also bring an array of advantages like being able to connect with people in different parts of the world.
### Response:"""
    tok_ids = encode(prompt, verifier_name)
    tok_ids = await run(
        manager_cls=Manager,
        verifier_cls=VerifierSlow,
        drafter_cls=Drafter,
        verifier_name=verifier_name,
        drafter_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        vocab_size=128256,
        verifier_dtype=torch.float16,
        drafter_dtype=torch.float16,
        verifier_load_in_8bit=True,
        drafter_load_in_8bit=True,
        lookahead=5,
        tok_ids=tok_ids,
        max_new_tokens=100,
    )
    # tok_ids = generate(
    #     model_name=verifier_name,
    #     dtype=verifier_dtype,
    #     load_in_8bit=verifier_load_in_8bit,
    #     tok_ids=tok_ids,
    #     max_new_tokens=max_new_tokens,
    # )
    print(f"Main: Final output: {decode(tok_ids, verifier_name)}")


if __name__ == "__main__":
    print("Starting script. Starting CUDA memory recording.")
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    try:
        asyncio.run(main())
        print("Shutting down all asyncio tasks")
        # Get the current event loop
        loop = asyncio.get_event_loop()
        # Cancel all remaining tasks
        tasks = asyncio.all_tasks(loop)
        for task in tasks:
            task.cancel()
        # Wait for all tasks to complete
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        # Shutdown asynchronous generators
        loop.run_until_complete(loop.shutdown_asyncgens())
        # Close the loop
        loop.close()
    except Exception as e:
        print(f"Exception occurred while running asyncio tasks or shutting them down: {e}")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"cuda_memory_snapshot_{current_time}.pickle"
    print(f"Dumping CUDA memory snapshot into {filename}.")
    torch.cuda.memory._dump_snapshot(filename)
    print(f"CUDA memory snapshot dumped into {filename}.")
    print("Script completed")
