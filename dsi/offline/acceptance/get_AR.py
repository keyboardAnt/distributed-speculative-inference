import tqdm
import transformers

import warnings
import logging

import numpy as np
from tqdm import tqdm

from dsi.configs.configs_AR import ConfigAcceptanteRate
from dsi.online.latency.get_latencies import MeasureLatencies

import torch

warnings.simplefilter(action='ignore', category=FutureWarning)

log = logging.getLogger(__name__)

class MeasureAcceptanceRate(MeasureLatencies):
    """
    Measures the generation acceptance rate for a given model and dataset.
    """
    def __init__(self, **config):
        self.config = ConfigAcceptanteRate(**config)
    
    def run(self, 
            model: str,
            dataset: str,
            subset: str = None,
            split: str = "test",
            model_revision: str = None
            ) -> float:
        
        all_n_matches = []
        config = gen_kwargs["assistant_model"].generation_config
        config.num_assistant_tokens_schedule = "constant"
        config.num_assistant_tokens = gen_kwargs["max_new_tokens"]
        transformers.set_seed(self.config.seed)

        model, tokenizer = self.load_model_tokenizer(model, model_revision)
        gen_kwargs = dict(do_sample=False, temperature=1.0, top_p=1.0, pad_token_id=tokenizer.eos_token_id)
        prompted_examples = self.get_random_prompted_examples(dataset, subset, split)

        for inputs in enumerate(tqdm(prompted_examples)):
            n_matches = [0]
            output_target = model.generate(**inputs, do_sample=self.config.do_sample_target, max_new_tokens=self.config.max_new_tokens, 
                                           pad_token_id = tokenizer.eos_token_id)
            prompt_len = len(inputs.input_ids[0])

            for i in range(prompt_len, len(output_target.sequences[0])):
                inputs['input_ids'] = output_target.sequences[0,0:i].view(1, -1)
                inputs['attention_mask'] = torch.tensor([[1] * i], device=model.device)
                output_draft = model.assistant_model.generate(**inputs, do_sample=self.config.do_sample_target, max_new_tokens=1, 
                                                              pad_token_id=tokenizer.eos_token_id)
                position = i-prompt_len
                
                if output_draft[position] == output_target[position]:
                    n_matches[-1] += 1
                else:
                    n_matches.append(0)
            all_n_matches.append(n_matches)

        ar = 1 - ( 1 / (np.array(all_n_matches).mean()))
        print(f'Acceptance Rate : {round(ar*100,2)}')
        return ar