import logging
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import AutoModelForCausalLM

from dsi.online.actual.message import MsgVerifiedRightmost
from dsi.online.actual.state import State

log = logging.getLogger(__name__)


@dataclass
class SetupModel:
    gpu_id: int
    state: State
    _name: str


class Model:
    def __init__(self, setup: SetupModel) -> None:
        self.setup: SetupModel = setup
        self._model: AutoModelForCausalLM = self._get_model(setup._name)

    def _get_model(self, name: str) -> AutoModelForCausalLM:
        """Loads the model from the given name and moves it to the device."""
        cpu = "cpu"

        def get_device() -> str:
            """Returns the device of the model. Use CPU if GPU is not available, and add
            a warning."""
            nonlocal self
            if torch.cuda.device_count() > self.setup.gpu_id:
                return f"cuda:{self.setup.gpu_id}"
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
        log.debug("Drafting new tokens...")
        input_ids: Tensor = torch.tensor([self.setup.state.tok_ids], dtype=torch.int)
        log.debug(f"Input IDs: {input_ids}")
        index_first_draft = input_ids.shape[-1]
        log.debug(f"Index of first draft: {index_first_draft}")
        # TODO: Consider sampling instead of greedy decoding
        outputs: Tensor = self._model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False,
            return_dict=False,
        )
        log.debug(f"Outputs: {outputs}")
        tok_ids: list[int] = outputs[0][index_first_draft:].tolist()
        log.debug(f"Drafted tokens: {tok_ids}")
        self.setup.state.extend(tok_ids, verified=False)
        log.debug(f"State after drafting: {self.setup.state}")
        return tok_ids

    def verify(self, tok_ids: list[int]) -> MsgVerifiedRightmost:
        """
        Verifies the tokens and updates the state.
        Returns the token id and index of the last verified token.
        """
        with self.setup.state.lock:
            self.setup.state.extend(tok_ids, verified=False)
            logits: Tensor = self._get_logits()
            num_accepted: int = self._get_num_accepted(logits[:-1])
            self.setup.state.v += num_accepted
            self.setup.state.rollback(self.setup.state.v)
            tok_id_verified_rightmost: int = logits[num_accepted].argmax().item()
            self.setup.state.extend([tok_id_verified_rightmost], verified=True)
            return MsgVerifiedRightmost(
                v=self.setup.state.v,
                tok_id=tok_id_verified_rightmost,
            )

    def _get_logits(self) -> Tensor:
        """
        Computes a forward. Returns the logits corresponding to not-yet verified tokens.
        The number of returned logits is k+1, where k is the number of drafts.
        """
        input_ids: Tensor = torch.tensor([self.setup.state.tok_ids], dtype=torch.int)
        logits: Tensor = (
            self._model.forward(input_ids, output_hidden_states=True)
            .logits.detach()
            .squeeze()
        )
        return logits[self.setup.state.v :]

    def _get_num_accepted(self, logits: Tensor) -> int:
        """
        Returns the number of draft tokens accepted based on an exact match.
        We only accept tokens that match the argmax tokens up to the first mismatch.
        """
        with self.setup.state.lock:
            tok_ids: Tensor = torch.tensor(
                self.setup.state.tok_ids[self.setup.state.v + 1 :], dtype=torch.int
            )
        pred_ids: Tensor = logits.argmax(dim=-1)
        mismatches: Tensor = (tok_ids != pred_ids).nonzero(as_tuple=True)[0]
        if mismatches.numel() == 0:
            return len(tok_ids)
        leftmost_mismatch: int = mismatches.min().item()
        return leftmost_mismatch
