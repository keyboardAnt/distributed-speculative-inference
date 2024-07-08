from pydantic import BaseModel, Field


class _Config(BaseModel):
    """
    _Config is the base class for all configurations.
    """

    random_seed: int = 42
    num_repeats: int = Field(5, title="The number of times to repeat the run", ge=1)
