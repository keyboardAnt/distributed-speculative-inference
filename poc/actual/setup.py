import torch
from poc.actual.manager import Manager
from poc.actual.utils import get_device_map_filepath, load_device_map, print_gpu_memory
from poc.actual.worker import Drafter, Verifier, get_workers


import asyncio
from typing import Type


async def setup(
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
) -> tuple[Manager, list[Verifier], Drafter]:
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
    return manager, verifiers, drafter