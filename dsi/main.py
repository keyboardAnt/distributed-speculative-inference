"""The main entrypoint for the CLI."""

import logging
import os

import hydra
import pandas as pd
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from dsi.configs.config_cli import ConfigCLI, RunType
from dsi.offline.heatmap.objective import enrich
from dsi.offline.heatmap.ray_manager import RayManager
from dsi.offline.run.dsi import RunDSI
from dsi.offline.run.si import RunSI
from dsi.types.result import Result
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
    if cfg.run_type == RunType.offline:
        log.info("Running offline simulation")
        res_si: Result = RunSI(cfg.config_run).run()
        log.info("res_si: %s", res_si)
        res_dsi: Result = RunDSI(cfg.config_run).run()
        log.info("res_dsi: %s", res_dsi)
        log.info("Plotting SI")
        plot_si: PlotIters = PlotIters(
            result=res_si, suptitle=f"Latency of SI (lookahead={cfg.config_run.k})"
        )
        plot_si.plot()
    elif cfg.run_type == RunType.offline_heatmap:
        tmanager: RayManager = RayManager()
        df_heatmap: pd.DataFrame = tmanager.run()
        tmanager.store(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        log.info("df_heatmap.head():")
        log.info(df_heatmap.head())
        log.info("df_heatmap.describe():")
        log.info(df_heatmap.describe())
        enrich(df_heatmap)
    elif cfg.run_type == RunType.online:
        log.info(
            "Running online simulation."
            " Implementation with a thread pool to be added."
        )
        raise NotImplementedError
    else:
        raise ValueError(f"Invalid simulation type: {cfg.run_type}")
    plt.show()
    log.info("Done")
