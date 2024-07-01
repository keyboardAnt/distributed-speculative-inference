import warnings
import json
from collections import defaultdict
from time import perf_counter as time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm


warnings.simplefilter(action='ignore', category=FutureWarning)
torch.set_float32_matmul_precision('high')


def load_model(name, compiled=False):
    print(f"Loading: {name}...   {compiled=}")
    extra = {}
    if name == "SweatyCrayfish/llama-3-8b-quantized":
        pass
    elif name in ("LLMQ/LLaMA-3-8B-SmoothQuant-8bit-8bit",):
        pass
    else:
        extra['torch_dtype'] = torch.bfloat16
        if name == "lmsys/vicuna-7b-v1.3":
            extra["revision"] = "refs/pr/4"
        elif name == "lmsys/vicuna-13b-v1.3":
            extra["revision"] = "refs/pr/5"
        elif name == "bigcode/starcoder":
            extra["revision"] = "refs/pr/112"
        
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto", attn_implementation="flash_attention_2", **extra)
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = torch.compile(model) if compiled else model
    return model, tokenizer
    
DS_MEAN_INPUT_LENGTH = {}


def get_prompt(dataset, ex):
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
        # figure 12
        # prompt = f"""You are an expert Python programmer, and here is your task: {text}
        # Your code should pass these tests:\n\n{test_list}\nYour code should start with a [PYTHON] tag and end with a [/PYTHON] tag."""
        # figure 11
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


def get_mean_input_length(tok, path=None, name=None, field=None):
    query = (tok.__class__.__name__, path, name, field)
    cached_input_length = DS_MEAN_INPUT_LENGTH.get(query)
    if cached_input_length is not None:
        return cached_input_length
    else:
        ds = load_dataset(path=path, name=name, split="test")
        field_parts = field.split("+")
        tokens = tok(ds[field_parts[0]])["input_ids"]
        
        if len(field_parts) > 1:
            field_2_tokens = tok(ds[field_parts[1]])["input_ids"]
            tokens = [f1 + f2 for f1, f2 in zip(tokens, field_2_tokens)]
            
        mean_input_length = int(np.mean(list(map(len, tokens))))
        DS_MEAN_INPUT_LENGTH[query] = mean_input_length
        return mean_input_length


def get_random_input(length, device):
    input_ids = (torch.rand(length) * 100).to(int).view(1, -1).to(device)
    input_ids_plus = torch.tensor(6).view(1,-1).to(device)
    return input_ids, input_ids_plus


def get_fwd_time(model, input_ids, past_key_values=None):
    torch.cuda.synchronize()
    t = time()
    out = model(input_ids=input_ids, past_key_values=past_key_values)
    torch.cuda.synchronize()
    elapsed = time() - t
    return elapsed, out


def timed_generate(model, tokenizer, prompted_examples, max_new_tokens=256):
    gen_kwargs = dict(do_sample=False, temperature=1.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id)

    ttfts, tpots = [], []
    for run_i in range(1):
        warmup_run = False
        
        for i, ex in enumerate(tqdm(prompted_examples)):
            inputs = tokenizer(ex, return_tensors="pt").to(model.device)
    
            ttft, _ = get_fwd_time(model=model, input_ids=inputs["input_ids"])
            if not warmup_run:
                ttfts.append(ttft)
    
            t = time()
            outputs = model.generate(**inputs, **gen_kwargs, max_new_tokens=max_new_tokens)
            elapsed = time() - t

            if not warmup_run:
                input_len = inputs["input_ids"].shape[1]
                new_tokens = outputs.shape[1] - input_len
        
                elapsed_after_first = elapsed - ttft
                tpots.append(elapsed_after_first / new_tokens)

    return np.mean(ttfts) * 1000, np.mean(tpots) * 1000


vicuna_sanity = (
    "double7/vicuna-68m",
    "double7/vicuna-68m",
)
starcoder_sanity = (
    "bigcode/tiny_starcoder_py",
    "bigcode/tiny_starcoder_py",
)

starcoder_models = (
    "bigcode/tiny_starcoder_py",
    "bigcode/starcoder",
)

vicuna_models = (
    "double7/vicuna-68m",
    "lmsys/vicuna-7b-v1.3",
    "lmsys/vicuna-13b-v1.3",
)

llama_models = (
    "double7/vicuna-68m",
    "meta-llama/Meta-Llama-3-8B",
)

phi3_models = (
    "microsoft/Phi-3-mini-128k-instruct",
    "microsoft/Phi-3-medium-128k-instruct",
)

cnn_dm = {"path": "cnn_dailymail", "name": "2.0.0", "field": "article"}
alpaca = {"path": "danielkorat/alpaca", "field": "instruction+input"}
mbpp = {"path": "mbpp", "field": "text+code"}
humaneval = {"path": "openai_humaneval", "field": "prompt"}

code_tasks = [mbpp, humaneval]
text_tasks = [cnn_dm, alpaca]

datasets = [cnn_dm, alpaca, mbpp, humaneval]
all_tasks = datasets


pairs_to_ds = {
    
    phi3_models: all_tasks,

    vicuna_models: text_tasks,
    
    starcoder_models: code_tasks,
    
}

config = dict(
    num_ex = 50,
    max_new_tokens=20,
)

def get_random_prompted_examples(ds_kwargs, num_ex=30, seed=42):
    examples = load_dataset(path=ds_kwargs["path"], name=ds_kwargs.get("name"), split="test")\
        .shuffle(seed=seed).select(range(num_ex))
    
    prompted_examples = []
    for ex in tqdm(examples, desc=f"{ds_kwargs['path']}"):
        prompt = get_prompt(ds_kwargs["path"], ex)
        prompted_examples.append(prompt)
        
    return prompted_examples

DS_TO_EXAMPLES = {}
for ds_kwargs in datasets:
    DS_TO_EXAMPLES[ds_kwargs["path"]] = get_random_prompted_examples(ds_kwargs, num_ex=config["num_ex"])
    
    
model_config = {
    "compiled": False,
}
config.pop("num_ex", None)

for model_family, ds_list in pairs_to_ds.items():
    model_family_res = defaultdict(list)
    for model_name in model_family:
        model, tokenizer = load_model(model_name, **model_config)

        # Warmup
        timed_generate(model, tokenizer, DS_TO_EXAMPLES[ds_list[0]["path"]], **config)
        
        for ds_kwargs in ds_list:
            ds_name = ds_kwargs["path"]
            print(model_name, ds_name)
            prompted_examples = DS_TO_EXAMPLES[ds_name]

            ttft, tpot = timed_generate(model, tokenizer, prompted_examples, **config)
            print(f"{ttft=}\n{tpot=}\n")

            model_family_res[ds_name].append({"model_name": model_name, "ttft": ttft, "tpot": tpot})
    
    with open("latencies.jsonl", "a") as f:
        for ds, ds_res in model_family_res.items():
            draft_res = ds_res[0]
            for model_res in ds_res[1:]:
                f.write(json.dumps({"dataset": ds, 
                           "target_name": model_res["model_name"], "target_first": model_res["ttft"], "target_sub": model_res["tpot"], 
                           "draft_name": draft_res["model_name"], "draft_first": draft_res["ttft"], "draft_sub": draft_res["tpot"]}))
                f.write("\n")
                