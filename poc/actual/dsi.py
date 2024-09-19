import asyncio
import time

import accelerate
from poc.actual.manager import Manager
from poc.actual.nonsi_hf import generate
from poc.actual.setup import setup
from poc.actual.utils import (
    cuda_memory_recording,
    decode,
    encode,
    print_gpu_memory,
    shutdown_asyncio,
)
from poc.actual.worker import Drafter, VerifierSlow
import torch


async def run(manager: Manager):
    print("Start measuring time NOW and run manager.")
    time_start = time.time()
    await manager.run()
    time_end = time.time()
    print(
        f"Generation completed. Time taken: {time_end - time_start:.2f} seconds"
    )
    print(f"Output tok_ids: {manager.tok_ids}")
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
    manager, verifiers, drafter = await setup(
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
    tok_ids = await run(manager)

    # NOTE: Hugging Face Generate has a lower overhead than our DSI implementation.
    #       We can use this to measure the overhead of our DSI implementation.
    def nonsi_hf_generate_baseline() -> torch.Tensor:
        nonlocal verifier_name, verifier_dtype, verifier_load_in_8bit, tok_ids, max_new_tokens
        tok_ids = generate(
            model_name=verifier_name,
            dtype=verifier_dtype,
            load_in_8bit=verifier_load_in_8bit,
            tok_ids=tok_ids,
            max_new_tokens=max_new_tokens,
        )
        return tok_ids
    # tok_ids = nonsi_hf_generate_baseline()

    print(f"Main: Final output: {decode(tok_ids, verifier_name)}")


if __name__ == "__main__":
    print("Starting script.")
    with cuda_memory_recording():
        asyncio.run(main())
        shutdown_asyncio()
    print("Script completed")
