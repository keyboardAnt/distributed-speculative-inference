# Distributed Speculative Inference of LLMs

The code used in the paper "[Distributed Speculative Inference of Large Language Models](https://arxiv.org/abs/2405.14105)" (arXiv, May 2024).

## Installation

Either use the devcontainer (recommended) or

1. Install poetry ([official documentation](https://python-poetry.org/docs/#installation)).
2. Install this project's environment: `poetry install`

Then:

3. Activate poetry's virtual environment: `poetry shell`

## Run

- analytic simulations: `python -m dsi`
- analytic heatmap simulations: `python -m dsi run_type=analytic_heatmap`. (- it initializes [Ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html))
- thread pool simulations: `python -m dsi run_type=thread_pool`

[Hydra](https://hydra.cc/docs/intro/) manages the configuration (defined at `dsi/config.py`). For example,
- to set the drafter latency (`c`) to 5%: `python -m dsi config_run.c=.05`
- to set the acceptance rate (`a`) to 50%:
`python -m dsi config_run.a=.5`

For more sophisticated combinations of configurations, check out Hydra's documentation.

## Testing

- Run tests: `python -m pytest` (from the project root)