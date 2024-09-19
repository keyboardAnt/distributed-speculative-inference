import asyncio
import os
import time
from typing import Type

import accelerate
from poc.actual.manager import Manager
from poc.actual.nonsi_hf import generate
from poc.actual.utils import (
    cuda_memory_recording,
    decode,
    encode,
    get_device_map_filepath,
    load_device_map,
    print_gpu_memory,
    shutdown_asyncio,
)
from poc.actual.worker import Drafter, Verifier, VerifierSlow, get_workers
import torch


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
    device_map_filepath = get_device_map_filepath(
        "device_map_meta-llama_Meta-Llama-3.1-70B-Instruct_8bit_on_3A40_custom.json"
    )
    verifier_device_map = load_device_map(device_map_filepath)
    verifier_2_device_map = {k: v + 3 for k, v in verifier_device_map.items()}
    verifiers_device_maps = [verifier_device_map, verifier_2_device_map]
    for i, device_map in enumerate(verifiers_device_maps):
        print(f"Main: Verifier {i} device map: {device_map}")
    verifiers, drafter = await get_workers(
        verify_queue=verify_queue,
        draft_queue=draft_queue,
        response_queue=response_queue,
        pubsub=manager.pubsub,
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
    print("Starting script.")
    with cuda_memory_recording():
        asyncio.run(main())
        shutdown_asyncio()
    print("Script completed")
