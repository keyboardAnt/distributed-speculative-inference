
from dataclasses import Field
from typing import Dict, List, Tuple


class ConfigLatency:
    """Includes all the parameters needed for measuring the latencies 
    of different model pairs tied to different datasets.
    """

    # Models
    starcoder_models = (
        "bigcode/tiny_starcoder_py",
        "bigcode/starcoder",
    )
    vicuna_models = (
        "double7/vicuna-68m",
        "lmsys/vicuna-7b-v1.3",
        "lmsys/vicuna-13b-v1.3",
    )
    phi3_models = (
        "microsoft/Phi-3-mini-128k-instruct",
        "microsoft/Phi-3-medium-128k-instruct",
    )

    # Datasets
    cnn_dm = {"path": "cnn_dailymail", "name": "2.0.0", "field": "article"}
    alpaca = {"path": "danielkorat/alpaca", "field": "instruction+input"}
    mbpp = {"path": "mbpp", "field": "text+code"}
    humaneval = {"path": "openai_humaneval", "field": "prompt"}
    code_datasets = [mbpp, humaneval]
    text_datasets = [cnn_dm, alpaca]
    all_datasets = [cnn_dm, alpaca, mbpp, humaneval]

    pairs_to_ds: Dict[Tuple[str], List[Dict[str, str]]] = Field(
        {
            phi3_models: all_datasets,
            vicuna_models: text_datasets,
            starcoder_models: code_datasets,
        },
        title="Pairs tied to datasets"
    )

    seed: int = Field(42, title="Seed for reproducibility")
    num_ex: int = Field(50, title="Number of examples to use for each dataset")
    max_new_tokens: int = Field(20, title="Maximum number of new tokens to generate")
    compiled_model: bool = Field(False, title="Whether to compile the model")
    flash_attn_impl: str = Field("flash_attention_2", title="Flash attention implementation")     
    model_revision: Dict[str, str] = Field(
        {
            "lmsys/vicuna-7b-v1.3": "refs/pr/4",
            "lmsys/vicuna-13b-v1.3": "refs/pr/5",
            "bigcode/starcoder": "refs/pr/112",
        },
        title="Model revisions"
    )
    save_latencies: bool = Field(True, title="Whether to save the latencies to json")
    
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