import logging

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from dsi.online.actual.message import MsgVerifiedRightmost
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
        cpu = "cpu"

        def get_device() -> str:
            """Returns the device of the model. Use CPU if GPU is not available, and add
            a warning."""
            nonlocal self
            if torch.cuda.device_count() > self.gpu_id:
                return f"cuda:{self.gpu_id}"
            log.warning("GPU not available. Using CPU.")
            return cpu

        m = AutoModelForCausalLM.from_pretrained(name)
        device: str = get_device()
        if device != cpu:
            m.to(device)
        return m

    def draft(self, max_new_tokens: int) -> list[int]:
        """
        Generate drafts and updates the state.
        Returns the generated input ids.
        """
        if max_new_tokens <= 0:
            return []
        input_ids: Tensor = torch.tensor([self.state.tok_ids], dtype=torch.int)
        outputs: Tensor = self._model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            use_cache=False,
            return_dict=False,
        )
        tok_ids: list[int] = outputs[0][len(self.state.tok_ids) :].tolist()
        self.state.extend(tok_ids, verified=False)
        return tok_ids

    def verify(self, tok_ids: list[int]) -> MsgVerifiedRightmost:
        """
        Verifies the tokens and updates the state.
        Returns the token id and index of the last verified token.
        """
        self.state.extend(tok_ids, verified=False)
        logits: Tensor = self._get_logits()
        num_accepted: int = self._get_num_accepted(logits[:-1])
        self.state.v += num_accepted
        self.state.rollback(self.state.v)
        tok_id_verified_rightmost: int = logits[num_accepted].argmax().item()
        self.state.extend([tok_id_verified_rightmost], verified=True)
        return MsgVerifiedRightmost(
            v=self.state.v,
            tok_id=tok_id_verified_rightmost,
        )

    def _get_logits(self) -> Tensor:
        """
        Computes a forward. Returns the logits corresponding to not-yet verified tokens.
        The number of returned logits is k+1, where k is the number of drafts.
        """
        input_ids: Tensor = torch.tensor([self.state.tok_ids], dtype=torch.int)
        logits: Tensor = (
            self._model.forward(input_ids, output_hidden_states=True)
            .logits.detach()
            .squeeze()
        )
        return logits[self.state.v :]

    def _get_num_accepted(self, logits: Tensor) -> int:
        """
        Returns the number of draft tokens accepted based on an exact match.
        We only accept tokens that match the argmax tokens up to the first mismatch.
        """
        tok_ids: Tensor = torch.tensor(
            self.state.tok_ids[self.state.v + 1 :], dtype=torch.int
        )
        pred_ids: Tensor = logits.argmax(dim=-1)
        mismatches: Tensor = (tok_ids != pred_ids).nonzero(as_tuple=True)[0]
        if mismatches.numel() == 0:
            return len(tok_ids)
        leftmost_mismatch: int = mismatches.min().item()
        return leftmost_mismatch
