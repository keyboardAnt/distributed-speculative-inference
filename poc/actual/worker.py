from poc.actual.pubsub import PubSub
import torch
from poc.actual.utils import print_gpu_memory, set_hf_cache
from poc.actual.event import Request, Response
from transformers import AutoModelForCausalLM
import accelerate

import asyncio
import os
import threading
import time
from abc import ABC, abstractmethod
from typing import Tuple, Type


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
        pubsub (PubSub): PubSub for receiving preemption messages.
        worker_id (int): ID of this worker.
    """

    def __init__(
        self,
        queue: asyncio.Queue[Request],
        response_queue: asyncio.Queue[Response],
        pubsub: PubSub,
        worker_id: int,
    ):
        self.pubsub = pubsub
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
        self.ready.clear()
        self.timestamp_preemption = 0
        self.timestamp_request = 0
        print(
            f"{self.__class__.__name__}: Resetting timestamp_preemption and timestamp_request"
        )
        self.ready.set()
        print(f"{self.__class__.__name__}: Ready event set")

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
            print(
                f"{self.__class__.__name__}: Loading model {name} without specifying device map"
            )
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
            print(f"{self.__class__.__name__}: {self.model.hf_device_map=}")
        except Exception as e:
            print(f"{self.__class__.__name__}: Error checking hf_device_map: {e}")
        print_gpu_memory()
        print(f"{self.__class__.__name__}: {next(self.model.parameters()).device=}")

    async def cancel_task(self, task: asyncio.Task) -> None:
        print(f"{self.__class__.__name__} ({self.worker_id}): Cancelling task.")
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            print(f"{self.__class__.__name__} ({self.worker_id}): Task was cancelled")
            return
        except Exception as e:
            print(
                f"{self.__class__.__name__} ({self.worker_id}): Task had an exception: {e}"
            )
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
        print(
            f"{self.__class__.__name__} ({self.worker_id}): Starting to process tasks"
        )
        self.ready.set()  # Ensure the ready event is set when run starts
        while True:
            preempt_queue = await self.pubsub.subscribe(self.worker_id)
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
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Current task was preempted"
                    )
                else:
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): No current task to cancel"
                    )
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
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Current task was preempted"
                    )
                else:
                    response = current_task.result()
                    await self.response_queue.put(response)
                    print(
                        f"{self.__class__.__name__} ({self.worker_id}): Task {request.id} completed. Response enqueued."
                    )
            for task in pending:
                await self.cancel_task(task)
            print(
                f"{self.__class__.__name__} ({self.worker_id}): Cancelled pending tasks"
            )

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
        print(
            f"{self.__class__.__name__} ({self.worker_id}): Getting scores for task {request.id}"
        )
        device = next(self.model.parameters()).device
        tok_ids = request.tok_ids.to(device)
        # Run in executor (i.e., separate thread) to avoid blocking the event loop
        scores: torch.Tensor
        tok_ids: torch.Tensor
        scores, tok_ids = await self.forward(tok_ids, request.n)
        # Move scores and tok_ids to the CPU
        scores = scores.to("cpu")
        tok_ids = tok_ids.to("cpu")
        print(
            f"{self.__class__.__name__} ({self.worker_id}): Computed scores of shape {scores.shape}"
        )
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
            tok_ids = tok_ids[:, : -n + 1]
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
        oracle_tok_ids = torch.tensor(
            [
                [
                    128000,
                    39314,
                    374,
                    459,
                    7754,
                    430,
                    16964,
                    264,
                    3465,
                    11,
                    35526,
                    449,
                    459,
                    1988,
                    430,
                    5825,
                    4726,
                    2317,
                    13,
                    9842,
                    264,
                    2077,
                    430,
                    36001,
                    45695,
                    279,
                    1715,
                    627,
                    14711,
                    30151,
                    512,
                    23340,
                    279,
                    1495,
                    1139,
                    1403,
                    20406,
                    43743,
                    627,
                    14711,
                    5688,
                    512,
                    791,
                    16659,
                    649,
                    387,
                    3974,
                    477,
                    4200,
                    11,
                    719,
                    449,
                    279,
                    28522,
                    14691,
                    304,
                    1690,
                    5596,
                    315,
                    279,
                    1917,
                    11,
                    1690,
                    5220,
                    690,
                    3469,
                    369,
                    4200,
                    16659,
                    304,
                    2015,
                    311,
                    30437,
                    279,
                    9041,
                    315,
                    17563,
                    13,
                    21382,
                    16659,
                    1101,
                    4546,
                    459,
                    1358,
                    315,
                    22934,
                    1093,
                    1694,
                    3025,
                    311,
                    4667,
                    449,
                    1274,
                    304,
                    2204,
                    5596,
                    315,
                    279,
                    1917,
                    627,
                    14711,
                    6075,
                    25,
                    4815,
                    791,
                    16659,
                    649,
                    387,
                    3974,
                    477,
                    4200,
                    11,
                    719,
                    449,
                    279,
                    28522,
                    14691,
                    304,
                    1690,
                    5596,
                    315,
                    279,
                    1917,
                    11,
                    1690,
                    5220,
                    690,
                    3469,
                    369,
                    4200,
                    16659,
                    304,
                    2015,
                    311,
                    30437,
                    279,
                    9041,
                    315,
                    17563,
                    382,
                    34126,
                    16659,
                    1101,
                    4546,
                    459,
                    1358,
                    315,
                    22934,
                    1093,
                    1694,
                    3025,
                    311,
                    4667,
                    449,
                    1274,
                    304,
                    2204,
                    5596,
                    315,
                    279,
                    1917,
                    13,
                    128009,
                    128006,
                    78191,
                    128007,
                    271,
                    791,
                    16659,
                    649,
                    387,
                    3974,
                    477,
                    4200,
                    11,
                    719,
                    449,
                    279,
                    28522,
                    14691,
                    304,
                    1690,
                    5596,
                    315,
                    279,
                    1917,
                    11,
                    1690,
                    5220,
                    690,
                    3469,
                    369,
                    4200,
                    16659,
                    304,
                    2015,
                    311,
                    30437,
                    279,
                    9041,
                    315,
                    17563,
                    382,
                ]
            ]
        )
        idx_first_new_token = tok_ids.shape[1]
        ret_tok_ids = oracle_tok_ids[:, idx_first_new_token : idx_first_new_token + n]
        ret_scores = torch.zeros((1, n, self.model.config.vocab_size))
        return ret_scores, ret_tok_ids


async def get_workers(
    verify_queue: asyncio.Queue,
    draft_queue: asyncio.Queue,
    response_queue: asyncio.Queue,
    pubsub: PubSub,
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
) -> list[Worker]:
    """
    Setup the workers and the pubsub system. Returns the workers.
    """
    set_hf_cache()
    print_gpu_memory()
    print("Main: Creating server instances")
    drafter = drafter_cls(
        queue=draft_queue,
        response_queue=response_queue,
        pubsub=pubsub,
        worker_id=0,
    )
    print("Main: Created drafter")
    verifiers = [
        verifier_cls(
            queue=verify_queue,
            response_queue=response_queue,
            pubsub=pubsub,
            worker_id=i,
        )
        for i in range(1, num_verifiers + 1)
    ]
    print(f"Main: Created {len(verifiers)} verifiers")
    for i, (verifier, device_map) in enumerate(zip(verifiers, verifiers_device_maps)):
        print(f"Main: Loading verifier {i}")
        await verifier.load_model(
            verifier_name,
            dtype=verifier_dtype,
            # device_map="auto",
            # device_map="balanced_low_0",
            device_map=device_map,
            load_in_8bit=verifier_load_in_8bit,
            cache_dir=os.environ["TRANSFORMERS_CACHE"],
        )
        print(f"Main: Verifier {i} loaded")
        print_gpu_memory()
    print(f"Main: All {len(verifiers)} verifiers loaded")
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
    asyncio.create_task(pubsub.broadcast())
    print("Main: Started PubSub broadcast")
    # Wait for the PubSub system to be ready
    await pubsub.ready.wait()
    print("Main: PubSub system is ready")
    asyncio.create_task(drafter.run())
    print("Main: Drafter task created")
    for verifier in verifiers:
        asyncio.create_task(verifier.run())
    print("Main: Verifiers tasks created")
    # Wait for all workers to be ready
    await asyncio.gather(
        drafter.ready.wait(), *[verifier.ready.wait() for verifier in verifiers]
    )
    print("Main: All workers are ready")
    workers = verifiers + [drafter]
    return workers
