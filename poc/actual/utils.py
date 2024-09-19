import gc
import json
import torch


import os

from transformers import AutoTokenizer


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
