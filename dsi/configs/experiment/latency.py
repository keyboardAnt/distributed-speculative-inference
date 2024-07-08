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
    compile_model: bool = Field(False, title="Whether to torch.compile() the model")
    model: str = Field(title="The model to use for the experiment")
    revision: None | str = Field(None, title="The revision of the model to use")
    dataset: str = Field(title="The dataset to use for the experiment")
    subset: None | str = Field(None, title="The subset of the dataset to use")
    split: str = Field("test", title="The split of the dataset to use")
