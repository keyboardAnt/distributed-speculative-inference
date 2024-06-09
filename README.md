# Distributed Speculative Inference of LLMs

The code used in the [paper](https://arxiv.org/abs/2405.14105).

## Installation 

1. Install poetry ([official documentation](https://python-poetry.org/docs/#installation)).
2. Install this project's environment: `poetry install`
3. Activate poetry's virtual environment: `poetry shell`

## Run

- analytic simulations: `python -m dsi analytic`
- thread pool simulations: `python -m dsi thread_pool`