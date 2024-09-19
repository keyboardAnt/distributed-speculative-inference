# from poc.actual.pubsub import PubSub
# import torch
# from poc.actual.manager import Manager
# from poc.actual.utils import get_queues, get_verifiers_device_maps, print_gpu_memory
# from poc.actual.worker import Drafter, Verifier, get_workers

# import asyncio
# from typing import Type


# async def setup(
#     num_verifiers: int,
#     verifier_cls: Type[Verifier],
#     drafter_cls: Type[Drafter],
#     verifier_name: str,
#     drafter_name: str,
#     verifier_dtype: torch.dtype,
#     drafter_dtype: torch.dtype,
#     verifier_load_in_8bit: bool,
#     drafter_load_in_8bit: bool,
#     verify_queue: asyncio.Queue,
#     draft_queue: asyncio.Queue,
#     response_queue: asyncio.Queue,
#     pubsub: PubSub,
# ) -> tuple[list[Verifier], Drafter]:
#     verifier_device_map_filename = "device_map_meta-llama_Meta-Llama-3.1-70B-Instruct_8bit_on_3A40_custom.json"
#     verifiers_device_maps = get_verifiers_device_maps(verifier_device_map_filename)
#     verifiers, drafter = await get_workers(
#         verify_queue=verify_queue,
#         draft_queue=draft_queue,
#         response_queue=response_queue,
#         pubsub=pubsub,
#         verifier_cls=verifier_cls,
#         drafter_cls=drafter_cls,
#         verifier_name=verifier_name,
#         drafter_name=drafter_name,
#         verifier_dtype=verifier_dtype,
#         drafter_dtype=drafter_dtype,
#         verifier_load_in_8bit=verifier_load_in_8bit,
#         drafter_load_in_8bit=drafter_load_in_8bit,
#         num_verifiers=num_verifiers,
#         verifiers_device_maps=verifiers_device_maps,
#         drafter_device_map=None,
#     )
#     return verifiers, drafter