from pydantic import Field

from dsi.configs.experiment.base import _ConfigExperiment


class ConfigLatency(_ConfigExperiment):
    """Includes all the parameters needed for measuring the latencies
    of a (target, draft, dataset) triplet.
    """

    num_ex: int = Field(50, title="The number of examples per dataset", ge=1)
    max_new_tokens: int = Field(
        20, title="The maximum number of new tokens to generate", ge=1
    )
    compiled_model: bool = Field(
        False, title="Whether to torch.compile() the model", ge=1
    )
