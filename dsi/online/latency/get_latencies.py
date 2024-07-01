import warnings
import json
from collections import defaultdict
from time import perf_counter as time

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

from dsi.configs.config_latency import ConfigLatency, get_prompt


warnings.simplefilter(action='ignore', category=FutureWarning)
    

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


class GetLatencies:
    """
    Measures the latencies of different model pairs tied to different datasets.
    """
    def __init__(self):
        self.config = ConfigLatency()

    def load_model_tokenizer(self, name):
        print(f"Loading: {name}...   {self.config.compiled_model=}")
        extra = {"torch_dtype": torch.bfloat16}
        if name in self.config.model_revision:
            extra["revision"] = self.config.model_revision[name]
            
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            attn_implementation=self.config.flash_attn_impl,
            **extra
            )
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = torch.compile(model) if self.config.compiled_model else model
        return model, tokenizer

    def get_random_prompted_examples(self, ds_kwargs):
        examples = load_dataset(path=ds_kwargs["path"], name=ds_kwargs.get("name"), split="test")\
            .shuffle(seed=self.config.seed).select(range(self.config.num_ex))
        
        prompted_examples = []
        for ex in tqdm(examples, desc=f"{ds_kwargs['path']}"):
            prompt = get_prompt(ds_kwargs["path"], ex)
            prompted_examples.append(prompt)
            
        return prompted_examples

    def timed_generate(self, model, tokenizer, prompted_examples):
        gen_kwargs = dict(do_sample=False, temperature=1.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id)

        ttfts, tpots = [], []
        for run_i in range(2):
            warmup_run = run_i < 1
        
            for ex in tqdm(prompted_examples):
                inputs = tokenizer(ex, return_tensors="pt").to(model.device)
                ttft, _ = get_fwd_time(model=model, input_ids=inputs["input_ids"])
                if not warmup_run:
                    ttfts.append(ttft)
        
                t = time()
                outputs = model.generate(**inputs, **gen_kwargs, max_new_tokens=self.config.max_new_tokens)
                elapsed = time() - t

                if not warmup_run:
                    input_len = inputs["input_ids"].shape[1]
                    new_tokens = outputs.shape[1] - input_len
                    elapsed_after_first = elapsed - ttft
                    tpots.append(elapsed_after_first / new_tokens)

        return np.mean(ttfts) * 1000, np.mean(tpots) * 1000


    def run(self):
        ds_to_examples = {}
        for ds_kwargs in self.config.all_datasets:
            ds_to_examples[ds_kwargs["path"]] = self.get_random_prompted_examples(ds_kwargs)
        
        for model_family, ds_list in self.config.pairs_to_ds.items():
            model_family_res = defaultdict(list)
            for model_name in model_family:
                model, tokenizer = self.load_model_tokenizer(model_name)

                # Warmup
                self.timed_generate(model, tokenizer, ds_to_examples[ds_list[0]["path"]])
                
                for ds_kwargs in ds_list:
                    ds_name = ds_kwargs["path"]
                    print(model_name, ds_name)
                    prompted_examples = ds_to_examples[ds_name]

                    ttft, tpot = self.timed_generate(model, tokenizer, prompted_examples)
                    print(f"{ttft=}\n{tpot=}\n")

                    model_family_res[ds_name].append({"model_name": model_name, "ttft": ttft, "tpot": tpot})
            
            if self.config.save_latencies:
                with open("latencies.jsonl", "a") as f:
                    for ds, ds_res in model_family_res.items():
                        draft_res = ds_res[0]
                        for model_res in ds_res[1:]:
                            f.write(json.dumps({"dataset": ds, 
                                    "target_name": model_res["model_name"], "target_first": model_res["ttft"], "target_sub": model_res["tpot"], 
                                    "draft_name": draft_res["model_name"], "draft_first": draft_res["ttft"], "draft_sub": draft_res["tpot"]}))
                            f.write("\n")
                        
        return model_family_res
                    