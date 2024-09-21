import asyncio
import statistics
import time
import uuid

import accelerate
from dsi.online.latency.dataset import Dataset
from poc.actual.manager import Manager
from poc.actual.nonsi_hf import generate
from poc.actual.prompt import get_prompts
from poc.actual.pubsub import PubSub
from poc.actual.utils import (
    cuda_memory_recording,
    decode,
    encode,
    garbage_collect,
    get_queues,
    get_verifiers_device_maps,
    print_gpu_memory,
    shutdown_asyncio,
)
from poc.actual.worker import Drafter, VerifierSlow, Worker, get_workers
import torch
from tqdm import tqdm


def cleanup():
    print("CLEANING UP: Syncing, collecting garbage, and printing GPU memory...")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    garbage_collect()
    print_gpu_memory()


async def get_latency(async_func, *args, **kwargs):
    print("Start measuring time NOW.")
    time_start = time.time()
    ret = await async_func(*args, **kwargs)
    time_end = time.time()
    latency = time_end - time_start
    print(f"Time taken: {latency:.2f} seconds")
    return latency, ret


@torch.no_grad()
async def main():
    print("Main started")
    manager_cls = Manager
    verifier_cls = VerifierSlow
    drafter_cls = Drafter
    # verifier_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    verifier_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    verifier_load_in_8bit = True
    verifier_dtype = torch.float16
    num_verifiers = 2
    # max_new_tokens = 100
    max_new_tokens = 10
    verifier_device_map_filename = "device_map_meta-llama_Meta-Llama-3.1-70B-Instruct_8bit_on_3A40_custom.json"
    verifiers_device_maps = get_verifiers_device_maps(verifier_device_map_filename)
    pubsub = PubSub()
    print(f"Number of verifiers: {num_verifiers}")
    verify_queue, draft_queue, response_queue = get_queues(num_verifiers)
    print("Queues created")
    workers: list[Worker] = await get_workers(
        verify_queue=verify_queue,
        draft_queue=draft_queue,
        response_queue=response_queue,
        pubsub=pubsub,
        verifier_cls=verifier_cls,
        drafter_cls=drafter_cls,
        verifier_name=verifier_name,
        drafter_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
        verifier_dtype=verifier_dtype,
        drafter_dtype=torch.float16,
        verifier_load_in_8bit=verifier_load_in_8bit,
        drafter_load_in_8bit=True,
        num_verifiers=num_verifiers,
        verifiers_device_maps=verifiers_device_maps,
        drafter_device_map=None,
    )
#     prompt: str = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# ### Instruction:
# Break the text into two logical paragraphs.
# ### Input:
# The meetings can be live or virtual, but with the pandemic continuing in many parts of the world, many companies will opt for virtual meetings in order to minimize the spread of illness. Virtual meetings also bring an array of advantages like being able to connect with people in different parts of the world.
# ### Response:"""
#     prompts = [prompt]
    prompts = get_prompts(dataset=Dataset.ALPACA_SRC, split="train", num_examples=50, random_seed=42)
    
    # NOTE: Hugging Face Generate has a lower overhead than our DSI implementation.
    #       We can use this to measure the overhead of our DSI implementation.
    async def run_nonsi_hf() -> torch.Tensor:
        nonlocal verifier_name, verifier_dtype, verifier_load_in_8bit, tok_ids, max_new_tokens
        tok_ids = generate(
            model_name=verifier_name,
            dtype=verifier_dtype,
            load_in_8bit=verifier_load_in_8bit,
            tok_ids=tok_ids,
            max_new_tokens=max_new_tokens,
        )
        return tok_ids

    async def run_our_implementation():
        nonlocal manager
        await manager.run()
        return manager.tok_ids
    
    run_func = run_our_implementation
    latencies = []
    for prompt in tqdm(prompts, desc="Prompts"):
        tok_ids = encode(prompt, verifier_name)

        manager = manager_cls(
            draft_queue=draft_queue,
            verify_queue=verify_queue,
            response_queue=response_queue,
            pubsub=pubsub,
            tok_ids=tok_ids,
            max_new_tokens=max_new_tokens,
            vocab_size=128256,
            lookahead=5,
        )
        print(f"Main: Created {manager.__class__.__name__}")
        cleanup()
        latency, tok_ids = await get_latency(run_func)
        latencies.append(latency)
        print(f"Main: Output tok_ids: {tok_ids}")
        print(f"Main: Final output: {decode(tok_ids, verifier_name)}")
        for worker in workers:
            worker.reset()
    print(f"Latencies: {latencies}")
    print(f"Mean latency: {sum(latencies) / len(latencies):.2f} seconds")
    print(f"Standard deviation: {statistics.stdev(latencies):.2f} seconds")


if __name__ == "__main__":
    print("Starting script.")
    with cuda_memory_recording():
        asyncio.run(main())
        shutdown_asyncio()
    print("Script completed")
