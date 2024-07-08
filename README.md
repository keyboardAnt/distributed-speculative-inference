# Distributed Speculative Inference of LLMs

The code used in the paper "[Distributed Speculative Inference of Large Language Models](https://arxiv.org/abs/2405.14105)" (arXiv, May 2024).

The library includes four experiments:
1. Estimating the acceptance rate of off-the-shelf LLMs
2. Estimating the forward latency of off-the-shelf LLMs
3. Estimating the speedup of DSI (compared to SI and non-SI) by measuring wall time, based on 1 and 2
4. Estimating the speedup of DSI (compared to SI and non-SI) by measuring time units

## Installation

Either use the devcontainer (recommended) or

1. Install poetry ([official documentation](https://python-poetry.org/docs/#installation)).
2. Install this project's environment: `poetry install`

Then:

3. Activate poetry's virtual environment: `poetry shell`

## Running experiments

There are two types of runs: offline (measuring time units or acceptance rate) and online (measuring wall time).

- offline simulations: `python -m dsi`
- heatmap of offline simulations: `python -m dsi type=offline_heatmap`. (- it initializes [Ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html))
- online simulations (implemented with a thread pool): `python -m dsi type=online`

[Hydra](https://hydra.cc/docs/intro/) manages the configuration (defined at `dsi/config.py`). For example,
- to set the drafter latency (`c`) to 5%: `python -m dsi run.c=.05`
- to set the acceptance rate (`a`) to 50%:
`python -m dsi run.a=.5`

For help, use:
`python -m dsi --help`

For more sophisticated combinations of configurations, check out Hydra's documentation.

## Testing

[![Python tests](https://github.com/keyboardAnt/distributed-speculative-inference/actions/workflows/python-tests.yaml/badge.svg)](https://github.com/keyboardAnt/distributed-speculative-inference/actions/workflows/python-tests.yaml)

Run tests: `python -m pytest` (from the project root)

## Stored results

[DVC](https://dvc.org/doc) tracks raw results stored on Google Drive. To pull the result: `dvc pull`
