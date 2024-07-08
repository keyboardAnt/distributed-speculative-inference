from pydantic import BaseModel, Field


class ConfigLatency(BaseModel):
    """Includes all the parameters needed for measuring the latencies
    of a (target, draft, dataset) triplet.
    """

    seed: int = Field(42, title="The random seed for each experiment")
    num_ex: int = Field(50, title="The number of examples per dataset", ge=1)
    max_new_tokens: int = Field(
        20, title="The maximum number of new tokens to generate", ge=1
    )
    compiled_model: bool = Field(
        False, title="Whether to torch.compile() the model", ge=1
    )
