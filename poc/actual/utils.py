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
    if torch.cuda.is_available():
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
        # Cancel all remaining tasks
        tasks = asyncio.all_tasks()
        for task in tasks:
            task.cancel()
        # Get the current event loop
        loop = asyncio.get_event_loop()
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
def cuda_memory_recording(max_entries: int):
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


def get_verifiers_device_maps(filename: str) -> list[dict]:
    device_map_filepath = get_device_map_filepath(filename)
    verifier_device_map = load_device_map(device_map_filepath)
    verifier_2_device_map = {k: v + 3 for k, v in verifier_device_map.items()}
    verifiers_device_maps = [verifier_device_map, verifier_2_device_map]
    for i, device_map in enumerate(verifiers_device_maps):
        print(f"Verifier {i} device map: {device_map}")
    return verifiers_device_maps


def get_queues(
    num_verifiers: int,
) -> tuple[asyncio.Queue, asyncio.Queue, asyncio.Queue]:
    verify_queue = asyncio.Queue(maxsize=num_verifiers)
    draft_queue = asyncio.Queue(maxsize=1)
    response_queue = asyncio.Queue()
    return verify_queue, draft_queue, response_queue


def right_pad_like(array_to_pad: torch.Tensor, like: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Right pads the `array_to_pad` with -1s to be like the `like` tensor.
    The shape of the `like` tensor should be the same as the `array_to_pad` except for the `dim` dimension.
    The `dim` dimension of the `like` tensor should be at least as large as the `dim` dimension of the `array_to_pad`.
    """
    if array_to_pad.dim() != like.dim():
        raise ValueError(f"Tensors must have the same number of dimensions. Got {array_to_pad.dim()} and {like.dim()}")
    
    if dim < 0 or dim >= array_to_pad.dim():
        raise ValueError(f"Invalid dimension {dim}. Must be between 0 and {array_to_pad.dim() - 1}")
    
    for i in range(array_to_pad.dim()):
        if i != dim and array_to_pad.shape[i] != like.shape[i]:
            raise ValueError(f"Shapes must match in all dimensions except {dim}. Mismatch at dimension {i}: {array_to_pad.shape[i]} vs {like.shape[i]}")
    
    if array_to_pad.shape[dim] > like.shape[dim]:
        raise ValueError(f"The 'dim' dimension of array_to_pad ({array_to_pad.shape[dim]}) cannot be larger than that of 'like' ({like.shape[dim]})")
    
    n = array_to_pad.shape[dim]
    padded = torch.full_like(like, -1)
    padded.narrow(dim, 0, n).copy_(array_to_pad)
    return padded


def get_shorter_mask(mask_1d: torch.Tensor, n: int) -> torch.Tensor:
    """
    The mask is a 1-dimensional boolean tensor of False values, with a subsequence of True values.
    This function "shortens" the mask by keeping the first `n` True values and replacing the rest with False values.

    Raises an error if `n` is strictly greater than the number of True values in the mask.
    
    Examples:
    - if the mask is [False, True, True, True, False] and `n` is 0, the function will return [False, True, True, True, False].
    - if the mask is [False, True, True, True, False] and `n` is 1, the function will return [False, True, False, False, False].
    - if the mask is [False, True, True, True, False] and `n` is 2, the function will return [False, True, True, False, False].
    - if the mask is [False, True, True, True, False] and `n` is 3, the function will return [False, False, False, False, False].
    - if the mask is [False, True, True, True, False] and `n` is 4, the function will raise an error.
    - if the mask is [True, True, True, False, False] and `n` is 2, the function will return [True, True, False, False, False].
    """
    true_count = mask_1d.sum().item()  # Count the number of True values in the mask
    if n > true_count:
        raise ValueError(f"Cannot shorten the mask to keep {n} True values because there are only {true_count} True values in the mask")
    if n == true_count:
        return mask_1d

    new_mask = mask_1d.clone()  # Clone the original mask to avoid modifying it directly
    if n == 0:
        new_mask[:] = False
        return new_mask

    true_indices = (new_mask == True).nonzero(as_tuple=True)[0]
    if n < true_indices.size(0):
        new_mask[true_indices[n]:] = False  # Set all True values beyond the first `n` to False

    return new_mask
