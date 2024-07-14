from pydantic import Field
import torch

from dsi.configs.experiment.base import _ConfigExperiment

DTYPE_MAP = {
        "torch.float32": torch.float32,
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16
        }

class ConfigLatency(_ConfigExperiment):
    """Includes all the parameters needed for measuring the latencies
    of a (target, draft, dataset) triplet.
    """

    model: str = Field(title="The model to use for the experiment")
    dtype: str = Field("bfloat16", title="The dtype to use for the experiment")
    dataset: str = Field(title="The dataset to use for the experiment")
    num_examples: int = Field(50, title="The number of examples per dataset", ge=1)
    max_new_tokens: int = Field(
        20, title="The maximum number of new tokens to generate", ge=1
    )
    compile_model: bool = Field(False, title="Whether to torch.compile() the model")
    revision: None | str = Field(None, title="The revision of the model to use")
    subset: None | str = Field(None, title="The subset of the dataset to use")
    split: str = Field("test", title="The split of the dataset to use")
    
    def get_torch_dtype(self) -> torch.dtype:
        """Get the torch dtype from the string."""
        return DTYPE_MAP.get(self.dtype, torch.bfloat16)
