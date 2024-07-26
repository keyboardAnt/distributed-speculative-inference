from typing import Literal

import torch
from pydantic import Field

from dsi.configs.experiment.base import _ConfigExperiment
from dsi.configs.experiment.generation import ConfigGen


class ConfigLatency(_ConfigExperiment):
    """Includes all the parameters needed for measuring the latencies
    of a (target, draft, dataset) triplet.
    """

    model: str = Field(title="The model to use for the experiment")
    dataset: str = Field(title="The dataset to use for the experiment")
    config_gen: ConfigGen = Field(
        default_factory=ConfigGen, title="The generation configuration"
    )
    dtype: Literal["float32", "float16", "bfloat16"] = Field(
        "bfloat16", title="The dtype to use for the experiment"
    )
    num_examples: int = Field(50, title="The number of examples per dataset", ge=1)
    max_new_tokens: int = Field(
        20, title="The maximum number of new tokens to generate", ge=1
    )
    compile_model: bool = Field(False, title="Whether to torch.compile() the model")
    revision: None | str = Field(None, title="The revision of the model to use")
    subset: None | str = Field(None, title="The subset of the dataset to use")
    split: str = Field("test", title="The split of the dataset to use")

    def get_torch_dtype(self) -> torch.dtype:
        return eval(f"torch.{self.dtype}")
