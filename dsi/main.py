"""The main entrypoint for the CLI."""

import logging
import os

import hydra
import pandas as pd
from omegaconf import OmegaConf

from dsi.configs.cli import ConfigCLI, RunType
from dsi.offline.heatmap.objective import enrich_inplace
from dsi.offline.heatmap.ray_manager import RayManager
from dsi.offline.simul.dsi import SimulDSI
from dsi.offline.simul.si import SimulSI
from dsi.types.df_heatmap import DataFrameHeatmap
from dsi.types.result import ResultSimul
from dsi.vis.iters_dist import PlotIters
from dsi.vis.utils import savefig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_name="config")
def main(cfg: ConfigCLI) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info("Working directory: %s", os.getcwd())
    log.info(
        "Output directory: %s",
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
    )
    log.info("Running new experiments")
    match cfg.type:
        case RunType.offline:
            log.info("Running offline simulation")
            res_si: ResultSimul = SimulSI(cfg.simul).run()
            log.info("res_si: %s", res_si)
            res_dsi: ResultSimul = SimulDSI(cfg.simul).run()
            log.info("res_dsi: %s", res_dsi)
            log.info("Plotting SI")
            plot_si: PlotIters = PlotIters(
                result=res_si, suptitle=f"Latency of SI (lookahead={cfg.simul.k})"
            )
            plot_si.plot()
            filepath_plots: str = savefig(name="si_latency_and_iters_dist")
            log.info("Figure saved at %s", filepath_plots)
        case RunType.offline_heatmap:
            tmanager: RayManager = RayManager(cfg.heatmap)
            df_results: pd.DataFrame = tmanager.run()
            df_heatmap: DataFrameHeatmap = enrich_inplace(df_results)
            filepath: str = df_heatmap.store()
            log.info("Heatmap stored at %s", filepath)
            log.info("df_heatmap.head():")
            log.info(df_heatmap.head())
            log.info("df_heatmap.describe():")
            log.info(df_heatmap.describe())
        case RunType.online:
            log.info(
                "Running online simulation."
                " Implementation with a thread pool to be added."
            )
            raise NotImplementedError
        case _:
            raise NotImplementedError(f"Invalid simulation type: {cfg.type}")
