import logging

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM

from dsi.configs.experiment.acceptance import ConfigAcceptanteRate
from dsi.configs.experiment.generation import ConfigGen
from dsi.online.latency.experiment import ExperimentLatency

log = logging.getLogger(__name__)


class ExperimentAcceptanceRate(ExperimentLatency):
    """
    Measures the generation acceptance rate.
    """
    def __init__(self, 
                 config: ConfigAcceptanteRate,
                 gen_config: ConfigGen,
                 draft_gen_config: ConfigGen):
        self.config: ConfigAcceptanteRate
        super().__init__(config, gen_config)
        self.draft_gen_config: ConfigGen = draft_gen_config
    
    def _load_draft_model(self) -> tuple:
        log.info(
            f"Loading model: {self.config.draft_model}, \
                compile={self.config.draft_compile_model}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.config.draft_model,
            device_map="auto",
            torch_dtype=self.config.get_torch_draft_dtype(),
            revision=self.config.draft_revision,
        )
        model = torch.compile(model) if self.config.draft_compile_model else model
        return model
     
    def run(self) -> float:
        
        all_n_matches = []

        examples = self._get_random_prompted_examples()
        target_model, tokenizer = self._load_model_tokenizer()

        target_model, tokenizer = self._load_model_tokenizer()
        target_gen_kwargs = dict(
            do_sample=self.gen_config.do_sample,
            temperature=self.gen_config.temperature,
            top_p=self.gen_config.top_p,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=self.config.max_new_tokens
        )
              
        draft_model = self._load_draft_model()
        draft_gen_kwargs = dict(do_sample=self.draft_gen_config.do_sample,
            temperature=self.draft_gen_config.temperature,
            top_p=self.draft_gen_config.top_p,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=1)

        for ex in tqdm(examples):
            inputs = tokenizer(ex, return_tensors="pt").to(target_model.device)
            n_matches = [0]
            output_target = target_model.generate(**inputs, **target_gen_kwargs)
            prompt_len = len(inputs.input_ids[0])

            for i in range(prompt_len, len(output_target[0])):
                inputs['input_ids'] = output_target[0,0:i].view(1, -1)
                inputs['attention_mask'] = torch.tensor([[1] * i], 
                                                        device=draft_model.device)
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
    ar_config = ConfigAcceptanteRate(model="lmsys/vicuna-7b-v1.3",
                                draft_model="double7/vicuna-68m", 
                                dataset="cnn_dailymail", subset="2.0.0")
    target_gen_config = ConfigGen(do_sample=False, temperature=1.0, top_p=1.0)
    draft_gen_config = ConfigGen(do_sample=False, temperature=1.0, top_p=1.0)
    mar = ExperimentAcceptanceRate(config=ar_config, 
                                   gen_config=target_gen_config, 
                                   draft_gen_config=draft_gen_config)
    mar.run()

if __name__ == "__main__":
    main()