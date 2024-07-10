import tqdm
import transformers

import warnings
import logging

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from dsi.configs.configs_AR import ConfigAcceptanteRate
from dsi.online.latency.measure_latencies import MeasureLatencies, Dataset

from transformers import AutoModelForCausalLM

import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)

class MeasureAcceptanceRate(MeasureLatencies):
    """
    Measures the generation acceptance rate for a given model and dataset.
    """
    def __init__(self, **config):
        self.config = ConfigAcceptanteRate(**config)
    
    def load_model(self, name: str, revision: str | None=None):
        log.info(f"Loading model: {name}, compiled={self.config.compiled_model}")
        extra_kwargs = {"torch_dtype": torch.bfloat16, "revision": revision}

        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            **extra_kwargs
            )
        model = torch.compile(model) if self.config.compiled_model else model
        return model
     
    def run(self, 
            target_model: str,
            draft_model: str,
            dataset: str,
            dataset_subset: str = None,
            split: str = "test",
            target_model_revision: str = None,
            draft_model_revision: str = None
            ) -> float:
        
        all_n_matches = []
        transformers.set_seed(self.config.seed)

        target_model, tokenizer = self.load_model_tokenizer(target_model, target_model_revision)
        target_gen_kwargs = dict(do_sample=self.config.do_sample_target, max_new_tokens=self.config.max_new_tokens, temperature=self.config.temp_target, 
                                  pad_token_id=tokenizer.eos_token_id)
        
        draft_model = self.load_model(draft_model, draft_model_revision)
        draft_gen_kwargs = dict(do_sample=self.config.do_sample_draft, max_new_tokens=1, temperature=self.config.temp_draft, 
                                pad_token_id=tokenizer.eos_token_id)
        
        prompted_examples = self.get_random_prompted_examples(Dataset(dataset), dataset_subset, split)

        for ex in tqdm(prompted_examples):
            inputs = tokenizer(ex, return_tensors="pt").to(target_model.device)
            n_matches = [0]
            output_target = target_model.generate(**inputs, **target_gen_kwargs)
            prompt_len = len(inputs.input_ids[0])

            for i in range(prompt_len, len(output_target[0])):
                inputs['input_ids'] = output_target[0,0:i].view(1, -1)
                inputs['attention_mask'] = torch.tensor([[1] * i], device=draft_model.device)
                output_draft = draft_model.generate(**inputs, **draft_gen_kwargs)
                if output_draft[-1, i] == output_target[-1, i]:
                    n_matches[-1] += 1
                elif i <  len(output_target[0]) - 1: # new window
                    n_matches.append(0)
                else: # at the end, remove last window
                    n_matches.pop()
            all_n_matches += n_matches
        print(f"{all_n_matches=}")
        ar = 1 - ( 1 / (1+np.array(all_n_matches).mean()))
        print(f'Acceptance Rate : {round(ar*100,2)}')
        return ar

def main():
    mar = MeasureAcceptanceRate()
    mar.run(target_model="lmsys/vicuna-13b-v1.3",
                                draft_model="double7/vicuna-68m", 
                                dataset="cnn_dailymail", dataset_subset="2.0.0")

if __name__ == "__main__":
    main()