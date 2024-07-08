from pydantic import BaseModel


class _Config(BaseModel):
    """
    _Config is the base class for all configurations.
    """

    random_seed: int = 42
