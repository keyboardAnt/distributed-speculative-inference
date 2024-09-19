import asyncio
import contextlib
from functools import cache
import os
import time
from typing import Type
from datetime import datetime

import accelerate
from poc.actual.manager import Manager
from poc.actual.nonsi_hf import generate
from poc.actual.utils import (
    decode,
    encode,
    load_device_map,
    print_gpu_memory,
    setup_hf_cache,
)
from poc.actual.worker import Drafter, Verifier, VerifierSlow
import torch


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
    drafter = drafter_cls(
        queue=draft_queue, response_queue=response_queue, manager=manager, worker_id=0
    )
    print("Main: Created drafter")
    verifiers = [
        verifier_cls(
            queue=verify_queue,
            response_queue=response_queue,
            manager=manager,
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
    asyncio.create_task(manager.pubsub.broadcast())
    print("Main: Started PubSub broadcast")
    # Wait for the PubSub system to be ready
    await manager.pubsub.ready.wait()
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
    verifier_device_map = load_device_map(
        "/workspace/distributed-speculative-inference/poc/actual/device_maps/device_map_meta-llama_Meta-Llama-3.1-70B-Instruct_8bit_on_3A40_custom.json"
    )
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


@torch.no_grad()
async def main():
    print("Script started")
    print_gpu_memory()
    # verifier_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    verifier_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    verifier_load_in_8bit = True
    verifier_dtype = torch.float16
    max_new_tokens = 100
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
        verifier_dtype=verifier_dtype,
        drafter_dtype=torch.float16,
        verifier_load_in_8bit=verifier_load_in_8bit,
        drafter_load_in_8bit=True,
        lookahead=5,
        tok_ids=tok_ids,
        max_new_tokens=max_new_tokens,
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
        print(
            f"Exception occurred while running asyncio tasks or shutting them down: {e}"
        )
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirname = "./cuda_memory_snapshots"
    filename = f"cuda_memory_snapshot_{current_time}.pickle"
    filepath = os.path.join(dirname, filename)
    print(f"Dumping CUDA memory snapshot into {filepath}.")
    torch.cuda.memory._dump_snapshot(filepath)
    print(f"CUDA memory snapshot dumped into {filepath}.")
    print("Script completed")
