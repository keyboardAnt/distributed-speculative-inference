from pydantic import BaseModel, Field


class _ConfigExperiment(BaseModel):
    """
    _ConfigExperiment is the base class for all experiment configurations.
    """

    random_seed: int = 42
    num_repeats: int = Field(
        5,
        title="The number of times to repeat a single run",
        description="For example, an experiment can average over multiple single runs.",
        ge=1,
    )
