"""The main entrypoint for the CLI."""

import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import OmegaConf

from dsi.configs.cli import ConfigCLI, RunType
from dsi.offline.heatmap.objective import enrich_inplace
from dsi.offline.heatmap.ray_manager import RayManager
from dsi.offline.run.dsi import RunDSI
from dsi.offline.run.si import RunSI
from dsi.types.df_heatmap import DataFrameHeatmap
from dsi.types.result import Result
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
    if cfg.load_results is None:
        log.info("Running new experiments")
        if cfg.type == RunType.offline:
            log.info("Running offline simulation")
            res_si: Result = RunSI(cfg.run).run()
            log.info("res_si: %s", res_si)
            res_dsi: Result = RunDSI(cfg.run).run()
            log.info("res_dsi: %s", res_dsi)
            log.info("Plotting SI")
            plot_si: PlotIters = PlotIters(
                result=res_si, suptitle=f"Latency of SI (lookahead={cfg.run.k})"
            )
            plot_si.plot()
        elif cfg.type == RunType.offline_heatmap:
            tmanager: RayManager = RayManager(cfg.heatmap)
            df_results: pd.DataFrame = tmanager.run()
            df_heatmap: DataFrameHeatmap = enrich_inplace(df_results)
            filepath_heatmap: Path = (
                Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
                / "heatmap"
            ).with_suffix(".csv")
            df_heatmap.to_csv(str(filepath_heatmap))
            log.info("df_heatmap.head():")
            log.info(df_heatmap.head())
            log.info("df_heatmap.describe():")
            log.info(df_heatmap.describe())

        elif cfg.type == RunType.online:
            log.info(
                "Running online simulation."
                " Implementation with a thread pool to be added."
            )
            raise NotImplementedError
        else:
            raise NotImplementedError(f"Invalid simulation type: {cfg.type}")
    else:
        log.info(
            "Received a path to load existing results."
            " Visualizing them rather than running new experiments."
        )
        log.info("Loading results from %s", cfg.load_results)
        raise NotImplementedError
    filepath_plots: str = savefig(name="si_latency_and_iters_dist")
    log.info("Figure saved at %s", filepath_plots)
    log.info("Done")
