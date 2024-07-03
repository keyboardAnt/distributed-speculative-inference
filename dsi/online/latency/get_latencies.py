from typing import Tuple
import warnings
from time import perf_counter as time
import logging

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from dsi.configs.config_latency import ConfigLatency

warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)

def get_prompt(dataset, ex):
    """Get the input prompt for the given dataset and example."""
    if dataset == "samsum":
        prompt = ex["dialogue"].strip("\n")
        prompt = f"Summarize this dialog:\n{prompt}\n---\nSummary:\n"
    elif dataset == "cnn_dailymail":
        prompt = ex["article"].strip("\n")
        prompt = f"""Summarize:
{prompt}
Summary:
"""
    elif dataset == "mbpp":
        # prompt from https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/
        text = ex["text"].strip("\n")
        test_list = ex["test_list"]
        prompt = f"""[INST]Your task is to write a Python function to solve a programming problem.
The Python code must be between [PYTHON] and [/PYTHON] tags.
You are given one example test from which you can infere the function signature.
Problem: Write a Python function to get the unique elements of a list.
Test: assert get_unique_elements([1, 2, 3, 2, 1]) == [1, 2, 3]
[/INST]
[PYTHON]
def get_unique_elements(my_list):
return list(set(my_list))
[/PYTHON]
[INST] Problem: {text}
Test: {test_list[0]}
[/INST]"""

    elif dataset == "danielkorat/alpaca":
        template_inp = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{inp}

### Response:
"""
        template_no_inp = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""
        if ex["input"].strip() == "":
            prompt = template_no_inp.format(instruction=ex["instruction"])
        else:
            prompt = template_inp.format(instruction=ex["instruction"], inp=ex["input"])
    else:
        prompt = ex["prompt"].strip("\n")
    return prompt
    
    
def get_random_input(length, device):
    input_ids = (torch.rand(length) * 100).to(int).view(1, -1).to(device)
    input_ids_plus = torch.tensor(6).view(1,-1).to(device)
    return input_ids, input_ids_plus


def get_fwd_time(model, input_ids, past_key_values=None):
    """Get the forward time of a model, with or without `past_key_values`."""
    torch.cuda.synchronize()
    t = time()
    out = model(input_ids=input_ids, past_key_values=past_key_values)
    torch.cuda.synchronize()
    elapsed = time() - t
    return elapsed, out


class MeasureLatencies:
    """
    Measures the generation latency for a given model and dataset.
    """
    def __init__(self, **config):
        self.config = ConfigLatency(**config)

    def load_model_tokenizer(self, name, revision=None):
        log.info(f"Loading model: {name}, compiled={self.config.compiled_model}")
        extra_kwargs = {"torch_dtype": torch.bfloat16, "revision": revision}

        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            attn_implementation=self.config.flash_attn_impl,
            **extra_kwargs
            )
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = torch.compile(model) if self.config.compiled_model else model
        return model, tokenizer

    def get_random_prompted_examples(self, dataset, subset=None, split="test"):
        """Get random examples from the dataset and prompt them."""
        log.info(f"Loading dataset: {dataset}, compiled={self.config.compiled_model}")
        examples = load_dataset(path=dataset, name=subset, split=split)\
            .shuffle(seed=self.config.seed).select(range(self.config.num_ex))
        return [get_prompt(dataset, ex) for ex in examples]

    def timed_generate(self, model, tokenizer, prompted_examples) -> Tuple[float, float]:
        """Time the generation of the prompted examples, distinguishing between
        the time to first token (ttft) and the time per output token (tpot)."""
        gen_kwargs = dict(do_sample=False, temperature=1.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id)
        ttfts, tpots = [], []

        for ex in tqdm(prompted_examples):
            # Get the time to first token by timing the forward pass
            inputs = tokenizer(ex, return_tensors="pt").to(model.device)
            ttft, _ = get_fwd_time(model=model, input_ids=inputs["input_ids"])
            ttfts.append(ttft)
            
            t = time()
            outputs = model.generate(**inputs, **gen_kwargs, max_new_tokens=self.config.max_new_tokens)
            elapsed = time() - t
            
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs.shape[1] - input_len
            elapsed_after_first = elapsed - ttft
            tpots.append(elapsed_after_first / new_tokens)
            
        return np.mean(ttfts) * 1000, np.mean(tpots) * 1000

    def run(self, 
            model: str,
            dataset: str,
            subset: str = None,
            split: str = "test",
            model_revision: str = None
            ) -> Tuple[float, float]:
        """Run the latency measurements for the given model and dataset."""
        examples = self.get_random_prompted_examples(dataset, subset, split)
        model, tokenizer = self.load_model_tokenizer(model, model_revision)

        # Warmup run
        self.timed_generate(model, tokenizer, examples)
        
        # Measure
        ttft, tpot = self.timed_generate(model, tokenizer, examples)
        log.info(f"{model=} {dataset=} {subset=}\n {ttft=} {tpot=}\n")
        return ttft, tpot
