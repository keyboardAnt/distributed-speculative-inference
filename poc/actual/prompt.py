from datasets import load_dataset
from dsi.online.latency.dataset import Dataset
from dsi.online.latency.prompts import get_prompt

def get_prompts(dataset: str, split: str, num_examples: int, random_seed: int) -> list[str]:
    """Get random examples from the dataset and prompt them."""
    examples = (
        load_dataset(
            path=dataset,
            split=split,
        )
        .shuffle(seed=random_seed)
        .select(range(num_examples))
    )
    prompted_examples = [get_prompt(dataset, ex) for ex in examples]
    return prompted_examples
