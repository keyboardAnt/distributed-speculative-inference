"""The main entrypoint for the CLI."""

import logging
import os

import hydra
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from dsi.analytic.si import RunSI
from dsi.config import Config, SimulationType
from dsi.schemas.results import ResultSI
from dsi.vis.iters_dist import PlotIters

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(cfg: Config) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Working directory: %s", os.getcwd())
    log.info(
        "Output directory: %s",
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    if cfg.simulation_type == SimulationType.analytic:
        res_si: ResultSI = RunSI(cfg.config_run).run()
        plot_static: PlotIters = PlotIters(
            result=res_si, suptitle=f"Latency of SI (lookahead={cfg.config_run.k})"
        )
        plot_static.plot()
    elif cfg.simulation_type == SimulationType.thread_pool:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid simulation type: {cfg.simulation_type}")
    plt.show()
    log.info("Done")
