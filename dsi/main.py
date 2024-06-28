"""The main entrypoint for the CLI."""

import logging
import os

import hydra
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from dsi.analytic.heatmap.ray_manager import RayManager
from dsi.analytic.si import RunSI
from dsi.configs.config_cli import ConfigCLI, RunType
from dsi.types.result import ResultSI
from dsi.vis.iters_dist import PlotIters

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(cfg: ConfigCLI) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Working directory: %s", os.getcwd())
    log.info(
        "Output directory: %s",
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    if cfg.run_type == RunType.analytic:
        res_si: ResultSI = RunSI(cfg.config_run).run()
        plot_si: PlotIters = PlotIters(
            result=res_si, suptitle=f"Latency of SI (lookahead={cfg.config_run.k})"
        )
        plot_si.plot()
    elif cfg.run_type == RunType.analytic_heatmap:
        tmanager: RayManager = RayManager()
        df_heatmap: pd.DataFrame = tmanager.run()
        tmanager.store(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        log.info("df_heatmap.head():")
        log.info(df_heatmap.head())
        log.info("df_heatmap.describe():")
        log.info(df_heatmap.describe())
    elif cfg.run_type == RunType.thread_pool:
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid simulation type: {cfg.run_type}")
    plt.show()
    log.info("Done")
