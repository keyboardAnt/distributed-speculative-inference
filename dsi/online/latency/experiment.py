import logging
from time import perf_counter as time

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from dsi.configs.experiment.latency import ConfigLatency
from dsi.online.latency.prompts import get_prompt
from dsi.types.experiment import _Experiment
from dsi.types.result import ResultLatency

log = logging.getLogger(__name__)


def get_fwd_time(model, input_ids, past_key_values=None):
    """Get the forward time of a model, with or without `past_key_values`."""
    torch.cuda.synchronize()
    t = time()
    out = model(input_ids=input_ids, past_key_values=past_key_values)
    torch.cuda.synchronize()
    elapsed = time() - t
    return elapsed, out


class ExperimentLatency(_Experiment):
    """
    Measures the generation latency for a given model and dataset.
    """

    def __init__(self, config: ConfigLatency):
        self.config: ConfigLatency
        super().__init__(config)

    def _load_model_tokenizer(self, name: str, revision: str | None = None):
        log.info(f"Loading model: {name}, compiled={self.config.compiled_model}")
        extra_kwargs = {"torch_dtype": torch.bfloat16, "revision": revision}

        model = AutoModelForCausalLM.from_pretrained(
            name, device_map="auto", **extra_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = torch.compile(model) if self.config.compiled_model else model
        return model, tokenizer

    def _get_random_prompted_examples(self, dataset: str, subset=None, split="test"):
        """Get random examples from the dataset and prompt them."""
        log.info(f"Loading dataset: {dataset}, compiled={self.config.compiled_model}")
        examples = (
            load_dataset(path=dataset, name=subset, split=split)
            .shuffle(seed=self.config.random_seed)
            .select(range(self.config.num_ex))
        )
        prompted_examples = [get_prompt(dataset, ex) for ex in examples]
        return prompted_examples

    def _timed_generate(self, model, tokenizer, prompted_examples) -> ResultLatency:
        """Time the generation of the prompted examples, distinguishing between
        the time to first token (ttft) and the time per output token (tpot)."""
        gen_kwargs = dict(
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        ttfts, tpots = [], []

        for ex in tqdm(prompted_examples):
            # Get the time to first token by timing the forward pass
            inputs = tokenizer(ex, return_tensors="pt").to(model.device)
            ttft, _ = get_fwd_time(model=model, input_ids=inputs["input_ids"])
            ttfts.append(ttft)

            t = time()
            outputs = model.generate(
                **inputs, **gen_kwargs, max_new_tokens=self.config.max_new_tokens
            )
            elapsed = time() - t

            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs.shape[1] - input_len
            elapsed_after_first = elapsed - ttft
            tpots.append(elapsed_after_first / new_tokens)

        mean_ttft = np.mean(ttfts) * 1000
        mean_tpot = np.mean(tpots) * 1000
        return ResultLatency(ttft=[mean_ttft], tpot=[mean_tpot])

    def _get_empty_result(self) -> ResultLatency:
        return ResultLatency()

    def _single_repeat(
        self,
        model: str,
        dataset: str,
        subset: str = None,
        split: str = "test",
        model_revision: str = None,
    ) -> ResultLatency:
        """Run the latency measurements for the given model and dataset."""
        log.info(f"Running ExperimentLatency: {model=} {dataset=} {subset=}\n")
        examples = self._get_random_prompted_examples(dataset, subset, split)
        model, tokenizer = self._load_model_tokenizer(model, model_revision)

        # Warmup run
        self._timed_generate(model, tokenizer, examples)

        # Measure
        result = self._timed_generate(model, tokenizer, examples)
        return result
