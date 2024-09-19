import torch
from poc.actual.utils import print_gpu_memory, setup_hf_cache
from transformers import AutoModelForCausalLM


import os
import time


def generate(
    model_name: str,
    dtype: torch.dtype,
    load_in_8bit: bool,
    tok_ids: torch.Tensor,
    max_new_tokens: int,
) -> str:
    setup_hf_cache()
    print(f"Loading tokenizer for {model_name}")
    print_gpu_memory()
    print(f"Loading model {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        load_in_8bit=load_in_8bit,
        cache_dir=os.environ["TRANSFORMERS_CACHE"],
    )
    model.eval()
    print_gpu_memory()
    # model.to("cuda" if torch.cuda.is_available() else "cpu")
    time_start = time.time()
    device = next(model.parameters()).device
    tok_ids = tok_ids.to(device)
    outputs = model.generate(
        input_ids=tok_ids,
        attention_mask=torch.ones_like(tok_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=False,
        return_dict_in_generate=True,
        output_scores=False,
        output_logits=False,
        output_hidden_states=False,
        output_attentions=False,
    )
    time_end = time.time()
    print(
        f"Generating with model {model_name} took {time_end - time_start:.2f} seconds"
    )
    return outputs.sequences