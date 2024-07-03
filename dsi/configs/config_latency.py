from dataclasses import dataclass


@dataclass
class ConfigLatency:
    """Includes all the parameters needed for measuring the latencies 
    of a (target, draft, dataset) triplet.
    """

    seed: int = 42
    num_ex: int = 50
    max_new_tokens: int = 20
    compiled_model: bool = False
    flash_attn_impl: str = None