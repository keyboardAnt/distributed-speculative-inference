from dsi.online.latency.dataset import Dataset


def get_prompt(dataset: Dataset, ex) -> str:
    """Get the input prompt for the given dataset and example."""
    if dataset == Dataset.SAMSUM:
        prompt = ex["dialogue"].strip("\n")
        prompt = f"Summarize this dialog:\n{prompt}\n---\nSummary:\n"
    elif dataset == Dataset.CNN_DAILYMAIL:
        prompt = ex["article"].strip("\n")
        prompt = f"""Summarize:
{prompt}
Summary:
"""
    elif dataset == Dataset.MBPP:
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
[/INST]"""  # noqa: E501

    elif dataset == Dataset.ALPACA:
        template_inp = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{inp}

### Response:
"""  # noqa: E501
        template_no_inp = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
"""  # noqa: E501
        if ex["input"].strip() == "":
            prompt = template_no_inp.format(instruction=ex["instruction"])
        else:
            prompt = template_inp.format(instruction=ex["instruction"], inp=ex["input"])
    else:
        prompt = ex["prompt"].strip("\n")
    return prompt