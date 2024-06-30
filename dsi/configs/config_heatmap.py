from pydantic import BaseModel


class ConfigHeatmap(BaseModel):
    # TODO(#15): Support larger ndims with GPUs
    # ndim: int = 200
    ndim: int = 10
    c_min: float = 0.01
    a_min: float = 0.01
    a_max: float = 0.99
    k_step: int = 1
    num_target_servers: int = 7
