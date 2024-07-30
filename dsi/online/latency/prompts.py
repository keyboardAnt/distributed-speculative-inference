from dsi.online.latency.dataset import Dataset


def get_prompt(dataset: Dataset, example) -> str:
    """Get the input prompt for the given dataset and example."""
    if dataset == Dataset.CNN_DAILYMAIL:
        prompt = example["article"].strip("\n")
        prompt = f"""Summarize:
{prompt}
Summary:
"""
    elif dataset == Dataset.MBPP:
        # prompt from https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/
        text = example["text"].strip("\n")
        test_list = example["test_list"]
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
        if example["input"].strip() == "":
            prompt = template_no_inp.format(instruction=example["instruction"])
        else:
            prompt = template_inp.format(
                instruction=example["instruction"], inp=example["input"]
            )
    else:
        prompt = example["prompt"].strip("\n")
    return prompt
