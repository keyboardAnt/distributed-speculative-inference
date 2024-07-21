from typing import Literal

import torch
from pydantic import Field

from dsi.configs.experiment.latency import ConfigLatency


class ConfigAcceptanteRate(ConfigLatency):
    """Includes all the parameters needed for measuring the acceptance rate
    of a (target, draft, dataset) triplet.
    """

    draft_model: str = Field(title="The draft model to use for the experiment")
    draft_dtype: Literal["float32", "float16", "bfloat16"] = Field(
        "bfloat16", title="The dtype of the draft model to use for the experiment"
    )
    draft_compile_model: bool = Field(
        False, title="Whether to torch.compile() the draft model"
    )
    draft_revision: None | str = Field(
        None, title="The revision of the draft model to use"
    )

    def get_torch_draft_dtype(self) -> torch.dtype:
        return eval(f"torch.{self.draft_dtype}")
