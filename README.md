# Distributed Speculative Inference of LLMs

The code used in the [paper](https://arxiv.org/abs/2405.14105).

## Installation 

1. Install poetry ([official documentation](https://python-poetry.org/docs/#installation)).
2. Install this project's environment: `poetry install`

## Run

- analytic simulations: `poetry run python -m dsi analytic`
- thread pool simulations: `poetry run python -m dsi thread_pool`