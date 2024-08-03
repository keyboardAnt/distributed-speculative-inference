import logging

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from dsi.online.actual.state import State

log = logging.getLogger(__name__)


class Model:
    def __init__(self, gpu_id: int, name: str, is_verifier: bool, state: State) -> None:
        self.gpu_id: int = gpu_id
        self._model: AutoModelForCausalLM = self._get_model(name)
        self._is_verifier: bool = is_verifier
        self.state: State = state

    def _get_model(self, name: str) -> AutoModelForCausalLM:
        """Loads the model from the given name and moves it to the device."""

        def get_device() -> str:
            """Returns the device of the model. Use CPU if GPU is not available, and add
            a warning."""
            nonlocal self
            if torch.cuda.is_available():
                return f"cuda:{self.gpu_id}"
            log.warning("GPU not available. Using CPU.")
            return "cpu"

        device: str = get_device()
        m = AutoModelForCausalLM.from_pretrained(name)
        m.to(device)
        return m

    def generate(self, max_new_tokens: int) -> list[int]:
        """Returns the generated input ids."""
        input_ids: Tensor = torch.tensor([self.state.tok_ids], dtype=torch.int)
        outputs: Tensor = self._model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0][len(self.state.tok_ids) :].tolist()

    # TODO: implement this method
    # def verify(self, tok_ids: list[int]):
    #     raise NotImplementedError
