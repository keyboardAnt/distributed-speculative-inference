from dataclasses import dataclass


@dataclass
class ConfigAcceptanteRate:
    """Includes all the parameters needed for measuring the acceptance rate of a (target, draft, dataset) triplet.
    """
    seed: int = 42
    num_ex: int = 50
    max_new_tokens: int = 500
    compiled_model: bool = False
    flash_attn_impl: str = None
    do_sample_target: bool = False
    do_sample_draft: bool = False