<h1 align="center">
  DSI
</h1>

<h3 align="center">
The fastest lossless inference algorithm of LLMs
</h3>

<p align="center">
| <a href="https://arxiv.org/abs/2405.14105"><b>Paper</b></a> |
</p>

---

## About

Distributed Speculative Inference (DSI) is provably the fastest lossless inference algorithm, introduced in the paper "[Distributed Speculative Inference of Large Language Models](https://arxiv.org/abs/2405.14105)" (arXiv, May 2024). This repo includes an implementation of DSI and all four experiments from the paper:
1. Estimating the speedup of DSI (compared to SI and non-SI) by measuring wall time, based on 3 and 4
2. Estimating the speedup of DSI (compared to SI and non-SI) by counting forward passes
3. Estimating the acceptance rate of off-the-shelf LLMs
4. Estimating the forward latency of off-the-shelf LLMs



## Getting Started

#### Installation

Either use the devcontainer (recommended) or

1. Install poetry ([official documentation](https://python-poetry.org/docs/#installation)).
2. Install this project's environment: `poetry install`

Then

3. Activate poetry's virtual environment: `poetry shell`

#### Running experiments

There are two types of runs: offline (measuring time units or acceptance rate) and online (measuring wall time).

**Sanity check.** To run a sanity check with simple simulations:
```
python -m dsi
```

**Heatmaps.** To create a heatmap (like Figure 1 in the paper):
```
python -m dsi type=heatmap
```
The heatmap is essentially a grid of simulations with different configurations. Unless specified otherwise, the simulations are _offline_, namely, counting time units rather than wall time. You can control the resolution of the heatmap by setting its `ndim` parameter. The paper uses `ndim=20`:
```
python -m dsi type=heatmap heatmap.ndim=20
```
We use [Ray](https://docs.ray.io/en/latest/ray-core/walkthrough.html) to run offline simulations in parallel.

You can also compute a heatmap based on _online_ (rather than offline) simulations, namely, estimating wall time (rather than counting forward passes in time units):
```
python -m dsi type=heatmap heatmap.online=True
```
The online simulation of DSI uses a thread pool and estimates the wall time. Since the online simulations depend on the available resources, we run the simulations one by one. We do not use Ray for online simulations.

[Hydra](https://hydra.cc/docs/intro/) manages all the configurations (defined under `dsi/config`). For example,
- to set the drafter latency (`c`) to 5%: `python -m dsi simul.c=.05`
- to set the acceptance rate (`a`) to 50%:
`python -m dsi simul.a=.5`

To list all the configurable parameters:
```
python -m dsi --help
```

For more sophisticated combinations of configurations, check out Hydra's documentation.

#### Visualizing results

By default, running new experiments will also visualize the results. To visualize existing results (pre-computed), provide their path: `python -m dsi type=heatmap load_results="results/offline/heatmap/heatmap-20240702-012750.csv"`

#### Testing

[![Python tests](https://github.com/keyboardAnt/distributed-speculative-inference/actions/workflows/python-tests.yaml/badge.svg)](https://github.com/keyboardAnt/distributed-speculative-inference/actions/workflows/python-tests.yaml)

Run tests: `python -m pytest` (from the project root)

#### Formatting

Run `pre-commit run --all-files` to check formating and re-format when possible.

#### Stored results

[DVC](https://dvc.org/doc) tracks raw results stored on Google Drive. To pull the result: `dvc pull`

## Sponsors

Our efforts and resources are supported by the following organizations. Thank you for your support!

- Weizmann Institute
- Intel Labs


## Citation

If you use DSI (or the code in this repo) for your research, please cite our [paper](https://arxiv.org/abs/2405.14105):
```bibtex
@article{timor2024distributed,
  title={Distributed Speculative Inference of Large Language Models},
  author={Timor, Nadav and Mamou, Jonathan and Korat, Daniel and Berchansky, Moshe and Pereg, Oren and Wasserblat, Moshe and Galanti, Tomer and Gordon, Michal and Harel, David},
  journal={arXiv preprint arXiv:2405.14105},
  year={2024}
}
```
