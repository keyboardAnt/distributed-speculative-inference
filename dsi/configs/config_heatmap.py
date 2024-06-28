from pydantic import BaseModel


class ConfigHeatmap(BaseModel):
    # ndim: int = 200
    ndim: int = 10
    c_min: float = 0.01
    a_min: float = 0.01
    a_max: float = 0.99
    k_step: int = 1
