
from typing import Dict, List, Tuple

# Models
sanity = (
    "double7/vicuna-68m",
    "double7/vicuna-68m",
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


class ConfigLatency:
    """Includes all the parameters needed for measuring the latencies 
    of different model pairs tied to different datasets.
    """

    all_datasets: list = [cnn_dm, alpaca, mbpp, humaneval]

    pairs_to_ds: Dict[Tuple[str], List[Dict[str, str]]] = {
            phi3_models: all_datasets,
            vicuna_models: text_datasets,
            starcoder_models: code_datasets,
        }    

    seed: int = 42
    num_ex: int = 50
    max_new_tokens: int = 20
    compiled_model: bool = False
    flash_attn_impl: str = None
    model_revision: Dict[str, str] = {
            "lmsys/vicuna-7b-v1.3": "refs/pr/4",
            "lmsys/vicuna-13b-v1.3": "refs/pr/5",
            "bigcode/starcoder": "refs/pr/112",
        },
    
    save_latencies: bool = True
