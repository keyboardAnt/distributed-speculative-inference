import asyncio
import contextlib
from datetime import datetime
import gc
import json
import torch


import os

from transformers import AutoTokenizer


def set_hf_cache():
    if torch.cuda.device_count() > 0:
        os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache"
        os.environ["HF_HOME"] = "/workspace/hf_cache"
    print(
        f"Main: Set Hugging Face cache directory to {os.environ.get('TRANSFORMERS_CACHE', 'Not set')}"
    )
    print(
        f"Main: Set Hugging Face home directory to {os.environ.get('HF_HOME', 'Not set')}"
    )


def load_device_map(file_name: str):
    print(f"Loading device map from {file_name}")
    with open(file_name, "r") as f:
        device_map = json.load(f)
    return device_map


def garbage_collect():
    print("Collecting garbage...")
    gc.collect()
    torch.cuda.empty_cache()


def print_gpu_memory():
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


def shutdown_asyncio():
    try:
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


@contextlib.contextmanager
def cuda_memory_recording(max_entries=1_000_000):
    try:
        print("Starting CUDA memory recording.")
        torch.cuda.memory._record_memory_history(max_entries=max_entries)
        yield
    finally:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        dirname = os.path.join(current_dir, "cuda_memory_snapshots")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        filename = f"cuda_memory_snapshot_{current_time}.pickle"
        filepath = os.path.join(dirname, filename)
        print(f"Dumping CUDA memory snapshot into {filepath}.")
        torch.cuda.memory._dump_snapshot(filepath)
        print(f"CUDA memory snapshot dumped into {filepath}.")


def get_device_map_filepath(device_map_filename: str) -> str:
    current_filepath = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_filepath)
    device_map_dir = os.path.join(current_dir, "device_maps")
    return os.path.join(device_map_dir, device_map_filename)
